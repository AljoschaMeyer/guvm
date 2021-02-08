# Generic Unityped Virtual Machine

The generic unityped virtual machine (guvm) is a simple virtual machine for unityped (i.e. dynamically typed) programming languages. The virtual machine is generic over the type of values, a concrete vm is obtained by specifying the set of values, and some opaque functions that can be used to manipulate them. The instruction set provides everything else that is necessary: control flow, moving values in memory, and the ability to define and call functions.

The primary intent is not to provide an efficient implementation, but to offer precise, simple semantics, so that a programming language can be defined by specifying its value set and a translation of language constructs into guvm instructions.

The virtual machine does not attempt to be a compilation target for any currently popular languages, in fact it eschews some rather standard features in favor of simplicity: there is no exception handling, so invalid actions such as trying to invoke a non-function value immediately terminate program execution, and all functions have a fixed arity between zero and fifteen, invoking a function with the wrong number of arguments also terminates execution.

What the machine does provide are instructions for loading/storing values from/to memory, (conditionally) jumping to other instructions, defining functions (lexically scoped closures to be precise), calling closures and built-in functions, defining asynchronous closures, and concurrently calling asynchronous closures and asynchronous build-in functions. Concurrent calls use corporative, eventloop based scheduling, the execution model is single-threaded and fully deterministic.

## Preliminary Definitions

All definitions in the specification are given in a [rust](https://www.rust-lang.org/)y, typed pseudocode. The main difference to rust is that everything is moved around by value (as if every time implemented `Copy`). This is effectively regular mathematical notation, but using plaintext and with a well defined way of specifying algebraic datatypes. In this pseudocode, `Bool` denotes the type of truth values (`true` and `false`), `Nat` the type of natural numbers, `U64` the natural numbers representable with 64 bits, `U4` the natural numbers representable with four bits.

The expression `undefined!()` means that something must never occur: the semantics of the virtual machine only well defined on initial states for which no such expression is reachable. A state is also invalid if it contains different functions with the same ordinal (ignore this sentence on your first read-through).

Optional values are represented by the following type:

```rust
enum Option<A> {
  Some(A),
  None,
}

impl<A> Option<A> {
    fn unwrap(self) -> A {
        match self {
            Some(a) => a,
            None => undefined!(),
        }
    }
}
```

`Map<A, B>` is the type of partial mappings from instances of type `A` to instances of type `B`. Let `m: Map<A, B>`, `a: A`, `b: B`. Then `m.get(a): Option<B>` denotes `Some(v)`, where `v` is the value to which `m` maps `a`, or `None` if no such value exists. `m.insert(a, b): Map<A, B>` denotes an updated mapping that works just like `m` except that it maps `a` to `b`. If `m: Map<Nat, B>`, then `m.fresh_key(): Nat` denotes the least natural number that is not mapped by `m`.

`[A]` this is the type of finite sequences containing `A`s. Let `s: [A]`, `a: A`, `i: Nat`. Then `s.count(): Nat` denotes the number of elements in `s`. `s.push(a): [A]` denotes the sequence obtained by adding `a` as the new last element of `s`. `s.pop(): [A]` denotes the sequence obtained by removing the last element of `s` (undefined if `s` is the empty sequence). `s[i]: A` denotes the i-th element within `s` (0-based indexing, undefined if the sequence is too short). `s.last(): A` denotes `s[s.count() - 1]`, `[]` denotes the empty sequence. `s.replace(i, a): [A]` denotes the sequence obtained from `s` by replacing the i-th element with `a`, and is undefined if the sequence did not contain an i-th element.

We finally define some type synonyms that will help clarifying what certain index values are intended to be used for:

```rust
type ScopeId = Nat;
type InstructionIndex = Nat;
type GlobalIndex = Nat;
type LocalIndex = Nat;
type ScopeIndex = Nat;
type AncestorDistance = Nat;
type AsyncId = Nat;
type Arity = U4;
```

Whenever the definition of the virtual machine involves natural numbers, an implementation may substitute fixed with integers instead.

## Semantics

```rust
// The state of the virtual machine at a single point in time.
// This is generic over the type of values that the machine handles: it moves
// them around, computes intermediate ones, and ultimately either produces a
// single such value as the result of a computation, terminates erroneously, or
// runs without ever terminating.
pub struct VirtualMachine<V: Value> {
    // The instructions constitute the program that is being executed by the
    // vm. They never change during a run of the machine.
    instructions: [Instruction],

    // An index into the `instructions`, indicating which one to execute next.
    // Most instructions incremented this by one, thus sequentially going
    // through them. There are however some dedicated instructions for setting
    // the instruction counter to arbitrary values. Those values are however
    // always statically determined, there are no dynamically computed jumps.
    instruction_counter: InstructionIndex,

    // Locations containing values that are available for being accessed and
    // updated from any point in the program.
    globals: [V],

    // Locations containing values that are in principle available for being
    // accessed and updated from any point in the program, but only if the
    // correct `ScopeId` is available. This somewhat correspondends to
    // heap-allocated memory and is used for implementing lexically scoped
    // closures. A real implementation would perform garbage collection,
    // removing scopes whose id has become unreachable from the current state.
    scopes: Map<ScopeId, Scope>,

    // A stack that manages all the state for performing synchronous closure
    // calls. Calling a closure appends a StackFrame, returning from a closure
    // call pops the last StackFrame.
    //
    // Each frame stores some of bookkeeping information for knowing how to
    // continue when the call returns, provides some value locations that are
    // available only from within that call, and introduces a new Scope which
    // can be accessed from the call and from all invocations of closures that
    // are created from within the call.
    stack: [StackFrame<V>],

    // The asynchronous counterpart to the stack. Calling an asynchronous
    // closure creates an AsyncFrame and adds it to the map with a fresh id.
    // Just like StackFrames, AsyncFrames store (slightly different)
    // bookkeeping information, provide value locations local to the call, and
    // a new scope shared between the call and all closures created from within.
    //
    // When calling an asynchronous function, the current call continues
    // immediately without access to the result. A different part of the
    // closure's instructions, one that has access to the result, is scheduled
    // for being executed once the invoked asynchronous function has returned.
    // Conceptually, these two independent strands execute concurrently.
    // Importantly, both strands share the same AsyncFrame, so they have access
    // to the same values.
    //
    // The first strand belonging to a certain AsyncFrame to return a value
    // concludes that call. A strand can also *yield*, i.e. stopping without
    // returning. If all strands of a AsyncFrame yield, that call never
    // returns at all.
    //
    // Since strands can invoke asynchronous functions independently, the
    // asynchronous function calls do not form a stack and must instead be kept
    // in a map.
    asyncs: HashMap<AsyncId, AsyncFrame<V>>,

    // Since the AsyncFrames are unordered, the machine needs to keep track of
    // which one is currently being executed.
    current_async: AsyncId,

    // When invoking an asynchronous closure, this closure is not immediately
    // evaluated, rather the strand of the current closure that does not get
    // access to the return value continues immediately. Before that, a new
    // AsyncFrame is allocated for the new call, and the evaluated arguments
    // are placed in its local storage. Then, the identifier of the newly
    // created AsyncFrame and the index of the instruction at which execution
    // of that function should begin are pushed on this stack.
    //
    // When an asynchronous function returns, the execution immediately
    // continues at the place where it was invoked. Whenever an execution
    // strand of an asynchronous closure yields however, the newest entry on
    // this stack is popped and the indicated closure resumes execution.
    pending_stack: [(AsyncId, InstructionIndex)],

    // The event loop manages the parallel execution of built-in asynchronous
    // functions. Invoking a built-in asynchronous function does not
    // immediately perform a computation, it merely creates a `Future<V>`, which
    // is added to the event loop. This by itself does not cause anything to
    // happen, the loop merely stores a set of such futures. A Future<V> can be
    // polled by an event loop to eventually compute a `V`.
    //
    // When a strand of an asynchronous closure yields, and the pending_stack
    // is empty, all futures that have been added to the event loop start
    // executing in parallel. When the first of them has produced a value, the
    // asynchronous closure that performed the corresponding invocation of the
    // build-in asynchronous function continues execution. A more detailed
    // description of the functionality provided by the EventLoop is given
    // further below.
    //
    // If the event loop needs to produce a value but no future has been added
    // to it, execution of the whole program terminates erroneously.
    event_loop: EventLoop<V>,
}

// We now take a closer look at the way the virtual machine stores values.

// A Scope consists of a number of storage locations for values, and an
// optional reference to a parent scope. Each closure stores the reference of
// the scope in which it was created, or `None` if it wasn't created at
// runtime. Whenever a closure is invoked, a new scope for that invocation is
// created, whose parent scope is the one stored in the invoked closure.
enum Scope<V> {
    values: [V],
    parent: Option<ScopeId>,
}

// When a synchronous closure is invoked, a new StackFrame is created to hold
// the data necessary for managing that call. It stores the id of the Scope
// belonging to that call, and some values that can be accessed from the
// instructions that define what the closure does. When the call has produced a
// result, the machine looks up in the StackFrame where to store it, and at
// which instruction to continue execution.
struct StackFrame<V> {
    values: [V],
    scope: ScopeId,
    dst: Address,
    return_instruction: InstructionIndex,
}

// AsyncFrames work the same, but they additionally store the AsyncId of the
// invoking asynchronous closure, so that the machine can update the
// current_async when it returns.
//
// The triple of an address to store a return value at, and instruction to
// continue at and an AsyncId to continue as is called a Continuation.
struct Continuation {
    instruction: InstructionIndex,
    dst: Address,
    call: AsyncId,
}

struct AsyncFrame<V> {
    values: [V],
    scope: ScopeId,
    continuation: Continuation,
}

// The globals, scopes and frames are the only locations at which values can be
// stored. An Address can refer to these values.
enum Address {
    // An index into the sequence of global values.
    Global(GlobalIndex),
    // An index into the values stored by the frame of the currently executed
    // closure (synchronous or asynchronous).
    Local(LocalIndex),
    // A lookup this kind of address begins at the scope stars by the frame of
    // the current closure (synchronous or asynchronous), then traverses to the
    // parent scope `up` many times, and then indexes into the values of that
    // scope.
    Scoped { up: AncestorDistance, index: ScopeIndex },
}

// The local values that belong to the currently executing closure.
fn locals<V: Value>(vm: VirtualMachine<V>) -> [V] {
    // Check whether there is a synchronous closure being executed right now.
    match vm.stack.last() {
        // There is.
        Some(frame) => frame.values,
        // There isn't, so look up the current asynchronous frame.
        None => vm.asyncs[vm.current_async].values,
    }
}

// The scope that belongs to the currently executing closure.
fn scope<V: Value>(vm: VirtualMachine<V>) -> Scope<V> -> {
    // Check whether there is a synchronous closure being executed right now.
    match vm.stack.last() {
        // There is.
        Some(frame) => frame.scope,
        // There isn't, so look up the current asynchronous frame.
        None => vm.asyncs[vm.current_async].scope,
    }
}

// Load a value from an address.
fn load<V: Value>(a: Address, vm: VirtualMachine<V>) -> V {
    match a {
        Address::Global(i) => vm.globals[i],
        Address::Local(i) => locals(vm)[i],
        Address::Scoped { up, index } => load_scoped(scope(vm), up, index),
    }
}

// The recursive part of loading a value from a scope.
fn load_scoped<V: Value>(
    scope: Scope<V>,
    up: AncestorDistance,
    index: ScopeIndex,
) -> V {
    if up == 0 {
        scope.values[index]
    } else {
        load_scoped(scope.parent.unwrap(), up - 1, index)
    }
}
```

The documentation is a work in progress, see [here](https://github.com/AljoschaMeyer/guvm-rs/blob/49449dad129199f35f38419fd2829cc38e4997e6/src/lib.rs) for a simple reference implementation.








<!--

// Set a local value that belong to the currently executing closure.
fn set_local<V: Value>(i: LocalIndex, v: V, vm: VirtualMachine<V>) -> VirtualMachine<V> {
    match vm.stack.last() {
        Some(frame) => vm_update_stack_last(
            vm,
            stack_frame_update_locals(frame, frame.values.update(i, vc)),
        ),
        None => {
            let frame = vm.asyncs[vm.current_async];
            return vm_update_async(
                vm,
                vm.current_async,
                async_frame_update_locals(frame, frame.values.update(i, vc)),
            );
        }
    }
}




To define the virtual machine semantics, we need to introduce three different concepts: the set of runtime `Value`s, the `Instruction`s that are executed by the vm, and finally a vm `State`.

### Values

A `Value` is a runtime value of the vm:

```rust
enum Value {
  V(V),
  BuiltIn(B),
  Function(Function), // to be defined later
}
```

`V` can be chosen almost arbitrarily, the only constraints are that there is a function `V::truthy(v: V) -> Bool` that assigns to each `V` whether it triggers conditional jump instructions or not, and that there is a function `V::default() -> V` to initialize memory holding values.

`B` can also be chosen mostly arbitrarily, there are two constraints: there has to be a function `B::arity(b: B) -> Arity` specifying how many arguments the built-in function expects, and `B::invoke(b: B, b_state: S, arguments: [Value]) -> Option<(S, Value)>`, which fallibly computes a return value and updates the functions state. The precise workings are explained later when the semantics of the call instruction are given.

```rust
struct Function {
  // unique identifier for this value
  ordinal: Nat,
  // access to closed-over values
  scope: ScopeId,
  // the instructions that define the behavior of the function
  header: InstructionIndex,
}
```

Function values store three pieces of information: an *ordinal* which is guaranteed to be unique to that function value (it has no immediate effect on the virtual machine semantics, but many languages want to provide built-in functions that need to be able to uniquely identify function values), a reference to values that have been closed over, and a reference to the instructions that properly define the workings of the function.

### State

A `State` describes the full state of the virtual machine at a single point in time. The machine works by iteratively computing a new `State` from the previous one until a termination condition has been reached.

```rust
// the virtual machine state
enum State {
  // the instructions making up the program to be executed
  instruction_map: Map<InstructionIndex, Instruction>,
  // the instruction to execute next
  instruction_counter: InstructionIndex,

  // values that can be addressed from everywhere
  global_map: Map<GlobalIndex, Value>,

  // hierarchical collections of values that can only be addressed from
  // within certain functions
  scope_map: Map<ScopeId, Scope>,

  // the call stack
  stack: [StackFrame],

  // the states of stateful built-in functions
  built_in_states: Map<B, S>,
}

// a scope, giving access to some values and possibly a parent scope
struct Scope {
  // scopes form a tree
  parent: Option<ScopeId>,
  // values that can be accessed in this scope
  values: [Value],
}

// the data necessary for executing nested function calls
struct StackFrame {
  // where to continue program execution after the current function has returned
  return_instruction: InstructionIndex,
  // where to store the return value of the function
  dst: Address, // defined later
  // which scope can be accessed from the current function call
  scope: ScopeId,
  // values that can be accessed only from within the current function call
  values: [Value],
}
```

### Instructions

Instructions define the semantics of functions. Some of them interact with values, so they need to be able to address them:

```rust
// the different possibilities where to look up values
enum Address {
  // an address that can be resolved from everywhere
  Global(GlobalIndex),
  // an address that is local to a function call, located on the call stack
  Local(LocalIndex),
  // an address that can be resolved from only certain function calls
  Scoped { up: AncestorDistance, index: ScopeIndex },
}
```

There are seven different instructions:

```rust
// specifies how to compute the next state from the current one
enum Instruction {
  // jump to a specific instruction
  Jump(InstructionIndex),
  // jump to a specific instruction if the address holds a truthy value
  ConditionalJump { condition: Address, target: InstructionIndex },
  // copy the value from the first address to the second address
  Assign { srs: Address, dst: Address },
  // finish a function call, returning a value
  Return(Address),
  // marks the beginning of a function body
  FunctionHeader {
    arity: Arity,
    local: LocalIndex,
    scoped: ScopeIndex,
  },
  // create a new function value whose scope is a fresh child scope of
  // the current one
  CreateFunction { dst: Address, header: InstructionIndex },
  // call a function
  Call {
    dst: Address,
    callee: Address,
    arguments: [Address],
  },
}
```

### Executing Instructions

Many of the instructions need to load values from an address or to store the values at an address.

The semantics of loading the value at an address:

```rust
fn load(src: Address, s: State) -> Value {
  match src {
    Address::Global(i) => s.global_map.get(i),
    Address::Local(i) => s.stack.last().values[i],
    Address::Scoped { up, index } => load_scoped(s.stack.last().scope, s, up, index),
  }
}

fn load_scoped(
  scope: ScopeId,
  s: State,
  up: AncestorDistance,
  i: ScopeIndex,
) -> Value {
  if up == 0 {
    s.scope_map[scope].values[i]
  } else {
    match s.scope_map[scope].parent {
      Some(parent) => load_scoped(parent, s, up - 1, i),
      None => undefined!(),
    }
  }
}
```

The semantics of storing a value at an address:

```rust
fn store(v: Value, dst: Address, s: State) -> State {
  let mut new_s = s;
  match dst {
    Address::Global(i) => new_s.global_map.insert(i, v),
    Address::Local(i) => new_s.stack.last().values[i] = v,
    Address::Scoped { up, index } => new_s = store_scoped(v, s.stack.last().scope, s, up, index),
  };
  return new_s;
}

fn store_scoped(
  v: Value,
  scope: ScopeId,
  s: State,
  up: AncestorDistance,
  i: ScopeIndex,
) -> State {
  if up == 0 {
    let mut new_s = s;
    new_s.scope_map[scope].values[i] = v;
    return new_s;
  } else {
    match s.scope_map[scope].parent {
      Some(parent) => store_scoped(v, parent, s, up - 1, i),
      None => undefined!(),
    }
  }
}
```

With these operations defined, we can now specify the semantics of the virtual machine. The virtual machine operates by continuously computing a new `State` from an old one until it either successfully yields a `Value` or cannot continue.

```rust
// trying to go from one state to the next can have three different outcomes
enum Status {
  // there is no successor state because the computation finished
  Done(Value),
  // there is no successor state because something went wrong
  Nope,
  // computed the successor state
  Continue(State),
}

// compute the successor state by performing a single instruction
fn step(s: State) -> Status {
  // the successor state is a modified version of the old one
  let mut new_s = s;

  // perform the instruction indicated by the current instruction counter
  match s.instruction_map.get(s.instruction_counter) {
    // unconditional jump
    Instruction::Jump(target) => new_s.instruction_counter = target,

    // jump if the value at the condition address is truthy
    Instruction::ConditionalJump { condition, target } => {
      if V::truthy(load(condition, s)) {
        new_s.instruction_counter = target;
      } else {
        new_s.instruction_counter += 1;
      }
    }

    // load src and store it to dst
    Instruction::Assign { src, dst } => {
      new_s = store(load(src, s), dst, s);
      new_s.instruction_counter += 1;
    }

    // function headers are ignored during execution, they merely store data
    Instruction::FunctionHeader { arity, local, scoped } => new_s.instruction_counter += 1,

    // calling functions is the most complicated instruction
    Instruction::Call { dst, callee, arguments } => {
      // resolve the value to call
      let f = load(callee, s);

      // resolve the arguments
      let mut args = [];
      for (i, argument) in arguments.enumerate() {
        args = args.push(load(argument, s));
      }

      match f {
        // execution fails when trying to call a non-function value
        Value::V(v) => return Status::Nope,

        // calling a built-in function
        Value::BuiltIn(b) => {
          // execution fails if the wrong number of arguments is supplied
          if B::arity(b) != args.count() {
            return Status::Nope;
          } else {
            // retrieve the state corresponding to this built-in function
            let s = s.built_in_states.get(b);

            // delegate to the implementation of the built-in function
            match B::invoke(b, s, args) {
              // the implementation decided to halt the computation
              None => Status::Nope,
              // successfully computed a value and a new state for the function
              Some((new_s, v)) => {
                new_s = store(v, dst, s);
                new_s.built_in_states = s.built_in_states.insert(b, new_s);
                new_s.instruction_counter += 1;
              }
            }
          }
        }

        // calling a regular function
        Value::Function(Function { ordinal, scope, header }) => {
          // look up information from the function header
          match s.instruction_map.get(header) {
            Instruction::FunctionHeader { arity, local, scoped } => {
              // execution fails if the wrong number of arguments is supplied
              if arity != args.count() {
                return Status::Nope;
              } else {
                // prepare a new stack frame:
                // first, prepare the stack-allocated values
                let mut values = [];
                // allocate the correct number of values
                for _ in 0..local {
                  values = values.push(V::default());
                }
                // place the arguments in the new stack frame
                for (i, arg) in args.enumerate() {
                  values[i] = arg;
                }

                let frame = StackFrame {
                  return_instruction: s.instruction_counter + 1,
                  dst,
                  scope,
                  values,
                };

                new_s.stack = s.stack.push(frame);
                new_s.instruction_counter = header;
              }
            }

            // header must point to a function header instruction
            _ => undefined!(),
          }
        }
      }
    }

    // create a new function value
    CreateFunction { dst: Address, header: InstructionIndex } => {
      // look up information from the function header
      match s.instruction_map.get(header) {
        Instruction::FunctionHeader { arity, local, scoped } => {
          // choose an ordinal for the new function value
          let ordinal = magically_conjure_previously_unused_ordinal();

          // create a new scope for the function value
          let scope_id = s.scope_map.fresh_key();
          let mut values = [];
          for _ in 0..scoped {
            values = values.push(V::default());
          }
          let new_scope = Scope {
            parent: Some(s.stack.last().scope),
            values,
          }

          let f = Function {
            ordinal,
            scope: new_scope,
            header,
          };

          // store new function value to dst
          new_s = store(f, dst, s);

          // add the new scope to the state
          new_s.scope_map = s.scope_map.insert(scope_id, new_scope);

          new_s.instruction_counter += 1;
        }

        // header must point to a function header instruction
        _ => undefined!(),
      }
    }

    // end a function call, having computed the value stored at src
    Instruction::Return(src) => {
      // the returned value
      let v = load(src, s);

      // if there is no more call stack left, the vm has computed a value
      if s.stack.count() == 1 {
        return Status::Done(v);
      } else {
        // find out where to continue execution
        let StackFrame {
          return_instruction,
          dst,
          ..
        } = s.stack.last();

        // pop from the stack
        new_s.stack = s.stack.pop();

        // store the function result
        // note that the dst is resolved after popping
        new_s = store(v, dst, new_s);

        // continue execution from where the function was called
        new_s.instruction_counter = return_instruction;
      }
    }
  }

  return Status::Continue(new_s);
}

// the semantics of a virtual machine with initial state `c`
fn run(mut s: State) -> Option<Value> {
  while true {
    match step(s) {
      Status::Done(v) => return Some(v),
      Status::Nope => return None,
      Status::Continue(new_s) => c = new_s,
    }
  }
}
```

A note on determinism: `magically_conjure_previously_unused_ordinal(): Nat` is the only nondeterministic part of the specification. Practical implementations of dynamically typed languages often identify functions by some heap address, which is provided nondeterministically from an allocator. To obtain deterministic semantics, the virtual machine can keep a counter, incrementing it by one and then using the resulting number whenever a new ordinal is required. This version of the virtual machine is called the **deterministic generic unityped virtual machine** (dguvm).

## (Deterministic) Asynchronous Generic Unityped Virtual Machine

The (deterministic) asynchronous generic unityped virtual machine ((d)aguvm) is an extension of the (d)guvm that introduces asynchronous functions. Asynchronous functions are functions that might need to wait for some time until some outside condition allows them to resume. Multiple asynchronous functions can execute concurrently so that the waiting happens in parallel.

At any point, a program can call an asynchronous function with a regular `Call` instruction. This executes the function, idly waiting if necessary, and then resumes the program, storing the computed value at some destination address. For the rest of the program, this is indistinguishable from calling a synchronous function.

When already executing an asynchronous function, an asynchronous function invocation can also be performed concurrently via `ConcurrentCall` instruction. It is also possible to synchronously call an asynchronous function from within an asynchronous function, which would create a completely independent context for the new invocation. Asynchronous execution contexts can thus form a stack, and concurrent asynchronous function invocations are possible whenever that stack is nonempty.

In addition to the parameters `V`, `B` and `S` of the guvm, the aguvm also has a parameter `A`, the type of built-in asynchronous functions. The set of values is augmented accordingly:

```rust
enum Value {
  AsyncBuiltIn(A),
  V(V),
  BuiltIn(B),
  Function(Function),
}
```

`A` must provide a function `A::arity(a: A) -> Arity` specifying how many arguments the asynchronous built-in function expects, and `A::invoke(a: A, arguments: [Value]) -> Future<Option<Value>`, which fallibly computes a return value and as a side-effect may update the functions state. This computation does not have to complete immediately, it merely returns a `Future`, a handle to a value that will become available at some point.

The asynchronous virtual machine comes with a few new instructions, and modifies the function header:

```rust
// specifies how to compute the next state from the current one
enum Instruction {
  // new_guvm instructions

  // schedule a concurrent asynchronous function call,
  // then continuing execution of the current function
  ConcurrentCall {
    dst: Address,
    // where to continue execution when the called function has returned
    continue_at: InstructionIndex,
    callee: Address,
    arguments: [Address],
  },
  // start executing the next scheduled asynchronous function call
  Yield,

  // changed guvm instructions

  // marks the beginning of a function body
  FunctionHeader {
    asynchronous: Bool,
    arity: Arity,
    local: LocalIndex,
    scoped: ScopeIndex,
  },

  // unchanged guvm instructions

  // jump to a specific instruction
  Jump(InstructionIndex),
  // jump to a specific instruction if the address holds a truthy value
  ConditionalJump { condition: Address, target: InstructionIndex },
  // copy the value from the first address to the second address
  Assign { srs: Address, dst: Address },
  // finish a function call, returning a value
  Return(Address),
  // create a new function value whose scope is a fresh child scope of
  // the current one
  CreateFunction { dst: Address, header: InstructionIndex },
  // call a function
  Call {
    dst: Address,
    callee: Address,
    arguments: [Address],
  },
}
```

The state of an asynchronous virtual machine:

```rust
// identifies a currently running asynchronous function invocation
type AsyncCallId = Nat;

// the virtual machine state
enum State {
  // holds the states for currently executing asynchronous functions
  async_stack: [AsyncContext],

  // the call stack has been removed
  // stack: [StackFrame],

  // unchanged guvm state

  // the instructions making up the program to be executed
  instruction_map: Map<InstructionIndex, Instruction>,
  // the instruction to execute next
  instruction_counter: InstructionIndex,

  // values that can be addressed from everywhere
  global_map: Map<GlobalIndex, Value>,

  // hierarchical collections of values that can only be addressed from
  // within certain functions
  scope_map: Map<ScopeId, Scope>,

  // the states of stateful built-in functions
  built_in_states: Map<B, S>,
}

// the necessary state for concurrently executing asynchronous functions
struct AsyncContext {
  // the call stack for executing synchronous functions
  stack: [StackFrame],
  // all asynchronous calls that are currently awaiting completion
  active_calls: Map<AsyncCallId, AsyncCall>,
  // the call that is currently being executed
  current_call: AsyncCallId,
  // asynchronous calls to begin executing in the future
  pending_queue: [Continuation],
  // results of asynchronous calls are stored here
  return_queue: [AsyncReturn],
  // where to continue program execution after the context is done
  return_instruction: InstructionIndex,
  // where to store the return value of the context
  dst: Address,
}

// state associated with a single asynchronous function call
enum AsyncCall {
  Regular {
    // the scope accessible from this call
    scope: ScopeId,
    // local values accessible from this call
    values: [Value],
    // where to continue when this call returns
    // `None` indicates to return from the current asynchronous context
    continuation: Option<Continuation>,
  },
  BuiltIn(Continuation),
}

struct Continuation {
  // the instruction at which to continue execution
  instruction: InstructionIndex,
  // the call in which execution continues
  call: AsyncCallId,
}

// a computed value and the asynchronous call that produces it
struct AsyncReturn {
  // the value to return
  value: Value,
  // the call that returns it
  call: AsyncCallId,
}
```

The updated semantics:

```rust
// loading now needs to look at the async stack
fn load(src: Address, s: State) -> Value {
  match src {
    Address::Global(i) => s.global_map.get(i),
    Address::Local(i) => {
      let cx = s.async_stack.last();
      if cx.stack.count() == 0 {
        // within an asynchronous call
        return cx.active_calls.get(cx.current_call).values[i];
      } else {
        // within a synchronous call
        return cx.stack.last().values[i];
      }
    }
    Address::Scoped { up, index } => {
      let cx = s.async_stack.last();
      if cx.stack.count() == 0 {
        // within an asynchronous call
        return load_scoped(cx.active_calls.get(cx.current_call).scope, s, up, index);
      } else {
        // within a synchronous call
        return load_scoped(cx.stack.last().scope, s, up, index);
      }
    }
  }
}

// storing now needs to look at the async stack
fn store(v: Value, dst: Address, s: State) -> State {
  let mut new_s = s;
  match dst {
    Address::Global(i) => new_s.global_map.insert(i, v),
    Address::Local(i) => {
      let cx = s.async_stack.last();
      if cx.stack.count() == 0 {
        // within an asynchronous call
        cx.active_calls.get(cx.current_call).values[i] = v;
      } else {
        // within a synchronous call
        cx.stack.last().values[i] = v;
      }
    }
    Address::Scoped { up, index } => {
      let cx = s.async_stack.last();
      if cx.stack.count() == 0 {
        // within an asynchronous call
        new_s = store_scoped(v, cx.active_calls.get(cx.current_call).scope, s, up, index);
      } else {
        // within a synchronous call
        new_s = store_scoped(v, cx.stack.last().scope, s, up, index);
      }
    }
  };
  return new_s;
}

// compute the successor state by performing a single instruction
fn step(s: State) -> Status {
  let cx = s.async_stack.last();

  // the successor state is a modified version of the old one
  let mut new_s = s;

  // perform the instruction indicated by the current instruction counter
  match s.instruction_map.get(s.instruction_counter) {
    // new instructions

    // concurrently call an asynchronous function
    Instruction::ConcurrentCall {
      dst,
      continue_at,
      callee,
      arguments,
    } => {
      // undefined if executed from within a synchronous function call
      if cx.stack.count() > 0 {
        undefined!();
      }

      // resolve the value to call
      let f = load(callee, s);

      // resolve the arguments
      let mut args = [];
      for (i, argument) in arguments.enumerate() {
        args = args.push(load(argument, s));
      }

      match f {
        // execution fails when trying to call a non-asynchronous-function value
        Value::V(v) | Value::BuiltIn(_) => return Status::Nope,

        // concurrently calling a built-in asynchronous function
        Value::AsyncBuiltIn(a) => {
          // execution fails if the wrong number of arguments is supplied
          if A::arity(a) != args.count() {
            return Status::Nope;
          } else {
            // we assume that there is an event loop for each `AsyncContext`
            let event_loop = get_event_loop(cx);

            // a unique identifier for the call
            let call_id = cx.active_calls.fresh_key();

            // add as an active call
            let mut new_cx = cx;
            new_cx.active_calls = cx.active_calls.insert(call_id, AsyncCall::BuiltIn(Continuation {
              instruction: continue_at,
              call: cx.current_call,
            }));

            new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;

            // this call immediately returns, but starts executing the Future
            // in parallel to the vm execution, and also associates the call_id
            // with the resulting value
            event_loop.schedule_builtin(B::invoke(b, s, args), call_id);

            // update the instruction counter and continue execution
            new_s.instruction_counter = s.instruction_counter + 1;
          }
        }

        // concurrently calling a regular function
        Value::Function(Function { ordinal, scope, header }) => {
          // look up information from the function header
          match s.instruction_map.get(header) {
            Instruction::FunctionHeader { asynchronous, arity, local, scoped } => {
              if !asynchronous {
                // cannot concurrently call a synchronous function
                return Status::Nope;
              } else {
                // execution fails if the wrong number of arguments is supplied
                if arity != args.count() {
                  return Status::Nope;
                } else {
                  // first, prepare local values
                  let mut values = [];
                  // allocate the correct number of values
                  for _ in 0..local {
                    values = values.push(V::default());
                  }
                  // place the arguments in the new local values
                  for (i, arg) in args.enumerate() {
                    values[i] = arg;
                  }

                  let mut new_cx = cx;

                  // create a new AsyncCall
                  // a unique identifier for the call
                  let call_id = cx.active_calls.fresh_key();

                  // where to continue after it returns
                  let continuation = Continuation {
                    instruction: continue_at,
                    call: cx.current_call,
                  };

                  // add the call to the map of all currently active calls
                  new_cx.active_calls = new_cx.active_calls.insert(call_id, AsyncCall::Regular {
                    scope,
                    values,
                    continuation: Some(continuation),
                    children: Map::empty(),
                  });

                  // enqueue the call to be executed later
                  new_cx.pending_queue = cx.pending_queue.push(Continuation {
                    instruction: header,
                    call: call_id,
                  });

                  new_s.instruction_counter += 1;
                }
              }
            }

            // header must point to a function header instruction
            _ => undefined!(),
          }
        }
      }
    }

    // stop executing the currently active asynchronous call,
    // and move to the next enqueued one
    Instruction::Yield => {
      let mut new_cx = cx;

      // undefined if executed from within a synchronous function call
      if cx.stack.count() > 0 {
        undefined!();
      }

      // if a previous concurrent call has returned, use its result
      if new_cx.return_queue.count() > 0 {
        let AsyncReturn { value, call: call_id } = new_cx.return_queue.pop_first();

        match new_cx.active_calls.try_get(call_id) {
          // this asynchronous function call has already returned previously
          None => {
            new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;
            // by not changing any state, new_s.instruction_counter still points
            // to a yield instruction, so execution will continue as if that
             // result had never existed
          }
          Some(call) => {
            // mark this call as having returned
            new_cx.active_calls = new_cx.active_calls.remove(call.id)
            // advance to next state from which to continue execution
            match call.continuation {
              // the return value belongs to a concurrent call
              Some(Continuation { instruction, call }) => {
                new_cx.current_call = call;
                new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;
                new_s.instruction_counter = instruction;
              }
              // the return value belongs to a blocking call
              None => {
                if new_s.async_stack.count() == 0 {
                  // program has terminated
                  return Status::Done(value);
                } else {
                  // pop the async stack and continue execution
                  new_s.async_stack = new_s.async_stack.pop();
                  new_s = store(value, new_cx.dst, new_s);
                  new_s.instruction_counter = new_cx.return_instruction;
                }
              }
            }
          }
        }
      } else {
        // no concurrent calls have returned yet, start the next enqueued one
        if new_cx.pending_queue.count() > 0 {
          let Continuation { instruction, call } = new_cx.pending_queue.pop();
          new_cx.current_call = call;
          new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;
          new_s.instruction_counter = instruction;
        } else {
          // nothing pending

          let event_loop = get_event_loop(new_cx)
          if event_loop.has_pending_calls() {
            // block until a built-in asynchronous call has produced a value
            match event_loop.block_until_something_finishes() {
              None => return Status::Nope,
              Some(r) => new_cx.return_queue = new_cx.return_queue.push(r),
            }

            new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;

            // by not changing any state, new_s.instruction_counter still points
            // to a yield instruction, so execution will continue by using the
            // new result
          } else {
            // nothing pending anymore, the computation got stuck
            return Status::Nope;
          }
        }
      }
    }

    // the instructions whose semantics changed compared to the guvm

    // calling a function, blocking if it is asynchronous
    Instruction::Call { dst, callee, arguments } => {
      // resolve the value to call
      let f = load(callee, s);

      // resolve the arguments
      let mut args = [];
      for argument in arguments {
        args = args.push(load(argument, s));
      }

      match f {
        // execution fails when trying to call a non-function value
        Value::V(v) => return Status::Nope,

        // calling a built-in function
        Value::BuiltIn(b) => {
          // execution fails if the wrong number of arguments is supplied
          if B::arity(b) != args.count() {
            return Status::Nope;
          } else {
            // retrieve the state corresponding to this built-in function
            let s = s.built_in_states.get(b);

            // delegate to the implementation of the built-in function
            match B::invoke(b, s, args) {
              // the implementation decided to halt the computation
              None => Status::Nope,
              // successfully computed a value and a new state for the function
              Some((new_s, v)) => {
                new_s = store(v, dst, s);
                new_s.built_in_states = s.built_in_states.insert(b, new_s);
                new_s.instruction_counter += 1;
              }
            }
          }
        }

        // calling a built-in asynchronous function
        Value::AsyncBuiltIn(a) => {
          // execution fails if the wrong number of arguments is supplied
          if A::arity(a) != args.count() {
            return Status::Nope;
          } else {
            // delegate to the implementation of the built-in function,
            // blocking until the Future results to a value
            match B::invoke(b, args).block() {
              // the implementation decided to halt the computation
              None => Status::Nope,
              // successfully computed a value
              Some(v) => {
                new_s = store(v, dst, s);
                new_s.instruction_counter += 1;
              }
            }
          }
        }

        // calling a regular function
        Value::Function(Function { ordinal, scope, header }) => {
          // look up information from the function header
          match s.instruction_map.get(header) {
            Instruction::FunctionHeader { asynchronous, arity, local, scoped } => {
              // execution fails if the wrong number of arguments is supplied
              if arity != args.count() {
                return Status::Nope;
              } else {
                // first, prepare local values
                let mut values = [];
                // allocate the correct number of values
                for _ in 0..local {
                  values = values.push(V::default());
                }
                // place the arguments in the new local values
                for (i, arg) in args.enumerate() {
                  values[i] = arg;
                }

                if asynchronous {
                  // create a new async context
                  let mut new_cx = AsyncContext {
                    stack: [],
                    active_calls: Map::empty(),
                    current_call: 0,
                    pending_queue: [],
                    return_queue: [],
                    return_instruction: s.instruction_counter,
                    dst,
                  };

                  new_cx.active_calls = new_cx.active_calls.insert(0, AsyncCall::Regular {
                    scope,
                    values,
                    continuation: None,
                  });

                  new_s.async_stack = s.async_stack.push(new_cx);
                  new_s.instruction_counter = header;
                } else {
                  // calling a synchronous function
                  let frame = StackFrame {
                    return_instruction: s.instruction_counter + 1,
                    dst,
                    scope,
                    values,
                  };

                  new_s.async_stack.last().stack = cx.stack.push(frame);
                  new_s.instruction_counter = header;
                }
              }
            }

            // header must point to a function header instruction
            _ => undefined!(),
          }
        }
      }
    }

    // end a function call, having computed the value stored at src
    Instruction::Return(src) => {
      // the returned value
      let v = load(src, s);

      if cx.stack.count() > 0 {
        // returning from a synchronous call
        let mut new_cx = cx;

        // find out where to continue execution
        let StackFrame {
          return_instruction,
          dst,
          ..
        } = cx.stack.last();

        // pop from the stack
        new_cx.stack = cx.stack.pop();
        new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;

        // store the function result
        // note that the dst is resolved after popping
        new_s = store(v, dst, new_s);

        // continue execution from where the function was called
        new_s.instruction_counter = return_instruction;
      } else {
        // returning from an asynchronous call
        let mut new_cx = cx;

        new_cx.result_queue = cx.result_queue.push(AsyncReturn {
          value: v,
          call: cx.current_call,
        });

        new_s.async_stack[new_s.async_stack.count() - 1] = new_cx;
        // continue as if encountering a yield instruction
        new_s.instruction_counter = index_of_some_yield_instruction();
      }
    }

    // remaining instructions remain unchanged

    // create a new function value
    CreateFunction { dst: Address, header: InstructionIndex } => {
      // look up information from the function header
      match s.instruction_map.get(header) {
        Instruction::FunctionHeader { asynchronous, arity, local, scoped } => {
          // choose an ordinal for the new function value
          let ordinal = magically_conjure_previously_unused_ordinal();

          // create a new scope for the function value
          let scope_id = s.scope_map.fresh_key();
          let mut values = [];
          for _ in 0..scoped {
            values = values.push(V::default());
          }
          let new_scope = Scope {
            parent: Some(s.stack.last().scope),
            values,
          }

          let f = Function {
            ordinal,
            scope: new_scope,
            header,
          };

          // store new function value to dst
          new_s = store(f, dst, s);

          // add the new scope to the state
          new_s.scope_map = s.scope_map.insert(scope_id, new_scope);

          new_s.instruction_counter += 1;
        }

        // header must point to a function header instruction
        _ => undefined!(),
      }
    }

    // unconditional jump
    Instruction::Jump(target) => new_s.instruction_counter = target,

    // jump if the value at the condition address is truthy
    Instruction::ConditionalJump { condition, target } => {
      if V::truthy(load(condition, s)) {
        new_s.instruction_counter = target;
      } else {
        new_s.instruction_counter += 1;
      }
    }

    // load src and store it to dst
    Instruction::Assign { src, dst } => {
      new_s = store(load(src, s), dst, s);
      new_s.instruction_counter += 1;
    }

    // function headers are ignored during execution, they merely store data
    Instruction::FunctionHeader { asynchronous, arity, local, scoped } => new_s.instruction_counter += 1,
  }

  return Status::Continue(new_s);
}

// unchanged from the guvm

// the semantics of a virtual machine with initial state `c`
fn run(mut s: State) -> Option<Value> {
  while true {
    match step(s) {
      Status::Done(v) => return Some(v),
      Status::Nope => return None,
      Status::Continue(new_s) => c = new_s,
    }
  }
}

fn load_scoped(
  scope: ScopeId,
  s: State,
  up: AncestorDistance,
  i: ScopeIndex,
) -> Value {
  if up == 0 {
    s.scope_map[scope].values[i]
  } else {
    match s.scope_map[scope].parent {
      Some(parent) => load_scoped(parent, s, up - 1, i),
      None => undefined!(),
    }
  }
}

fn store_scoped(
  v: Value,
  scope: ScopeId,
  s: State,
  up: AncestorDistance,
  i: ScopeIndex,
) -> State {
  if up == 0 {
    let mut new_s = s;
    new_s.scope_map[scope].values[i] = v;
    return new_s;
  } else {
    match s.scope_map[scope].parent {
      Some(parent) => store_scoped(v, parent, s, up - 1, i),
      None => undefined!(),
    }
  }
}

// trying to go from one state to the next can have three different outcomes
enum Status {
  // there is no successor state because the computation finished
  Done(Value),
  // there is no successor state because something went wrong
  Nope,
  // computed the successor state
  Continue(State),
}
```




















Features that did not make it:
- modules, dynamic linking, dynamic loading, position independent code
- breakpoints, noop -->
