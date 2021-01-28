# Generic Unityped Virtual Machine

The generic unityped virtual machine (guvm) is a simple virtual machine for unityped (i.e. dynamically typed) programming languages whose definition is generic over the exact type of values. The language might support integers, arrays, whatevers, the guvm doesn't care. What it does provide are first-class function values (lexically scoped closures).

Functions work by executing instructions such as loading/storing values from/to memory, (conditionally) jumping to other instructions, or calling further functions. The semantics are kept fairly simple, there is no exception handling and every functions has a fixed arity.

This whole specification is generic over `V`, the type of language-specific non-function values, `B`, the type of built-in functions (see below), and `S`, the type of built-in function states (a rather technical necessity for giving the semantics of stateful built-in functions).

## Preliminary Definitions

All definitions in the specification are given in a [rust](https://www.rust-lang.org/)y, typed pseudocode. In this pseudocode, `Bool` denotes the type of truth values (`true` and `false`), `Nat` the type of natural numbers, `U64` the natural numbers representable with 64 bits.

Optional values are represented by the following type:

```rust
enum Option<A> {
  Some(A),
  None,
}
```

`Map<A, B>` is the type of partial mappings from instances of type `A` to instances of type `B`. Let `m: Map<A, B>`, `a: A`, `b: B`. Then `m.get(a): B` denotes the value to which `m` maps `a` (and is undefined if `m` does not map `a`). `m.insert(a, b): Map<A, B>` denotes an updated mapping that works just like `m` except that it maps `a` to `b`. If `m: Map<Nat, B>`, then `m.fresh_key(): Nat` denotes the least natural number that is not mapped by `m`.

`[A]` this is the type of finite sequences containing `A`s. Let `s: [A]`, `a: A`, `i: Nat`. Then `s.count(): Nat` denotes the number of elements in `s`, `s.push(a): [A]` denotes the sequence obtained by adding `a` as the new last element of `s`, `s.pop(): [A]` denotes the sequence obtained by removing the last element of `s` (undefined if `s` is the empty sequence) `s[i]` denotes the i-th element within `s` (0-based indexing, undefined if the sequence is too short), `s.last(): [A]` denotes `s[s.count() - 1]`, `[]` denotes the empty sequence.

We finally define some type synonyms that will help clarifying what certain index values are intended to be used for:

```rust
type ScopeId = Nat;
type InstructionIndex = Nat;
type GlobalIndex = Nat;
type LocalIndex = Nat;
type ScopeIndex = Nat;
type AncestorDistance = Nat;
type Arity = Nat;
```

Whenever the definition of the virtual machine involves natural numbers, an implementation may substitute a finite subset of the natural numbers.

The specification of the language semantics involves expressions that may be undefined, e.g. `m.get(a)` for a mapping `m` that does not map `a`. The specification assumes that prior to running the virtual machine, a conservative check has been performed that makes sure that undefined cases never arise. Since the virtual machine is fairly static (no arbitrarily-computed jumps, fixed numbers of arguments, stack space and heap space for functions), such a check can be performed in linear time. Alternatively, an implementation can forgo such checks, instead exhibiting undefined behavior in any such case. This is not recommended when running untrusted code, but it makes sense when the code is known to be produced from a trusted source, e.g. a compiler that only emits a valid code.

Another precondition for the semantics to make sense is that the ordinals of all functions in a given `State` are distinct.

## Semantics

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

The (deterministic) asynchronous generic unityped virtual machine ((d)aguvm) is an extension of the (d)guvm that introduces asynchronous functions. Asynchronous functions are functions that might need to wait for some time onto some outside condition allows them to resume. Multiple asynchronous functions can execute concurrently so that the waiting happens in parallel.

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
    // all asynchronous calls that originated from this one
    children: Map<AsyncCallId, ()>,
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

            // add it to the set of child calls of the current call
            let mut new_cx = cx;
            let current = cx.active_calls.get(cx.current_call);
            let mut updated_current = current;
            updated_current.children = current.children.insert(call_id, ());
            new_cx.active_calls = cx.active_calls.insert(cx.current_call, updated_current);

            // add as an active call
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

                  // add it to the set of child calls of the current call
                  let current = cx.active_calls.get(cx.current_call);
                  let mut updated_current = current;
                  updated_current.children = current.children.insert(call_id, ());
                  new_cx.active_calls = cx.active_calls.insert(cx.current_call, updated_current);

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

      // check whether any built-in asynchronous functions have produced a
      // value so far
      let event_loop = get_event_loop(cx);
      for built_in_return in event_loop.finished() {
        match built_in_return {
          None => return Status::Nope,
          Some(r) => new_cx.return_queue = new_cx.return_queue.push(r),
        }
      }

      // if a previous concurrent call has returned, use its result
      if new_cx.return_queue.count() > 0 {
        let AsyncReturn { value, call: call_id } = new_cx.return_queue.pop_first();
        let call = new_cx.active_calls.get(call_id);

        // recursively clean up all children of the call that returned
        // (remove itself and all children from pending_queue and result_queue)
        new_cx = clean_up_active_call(call_id, cx);

        // advance to next state from which to continue execution
        match call.continuation {
          // the return value belongs to a concurrent call
          Some(Continuation { instruction, call }) => {
            new_cx.current_call = call;
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
      } else {
        // no concurrent calls have returned yet, start the next enqueued one
        if new_cx.pending_queue.count() > 0 {
          let Continuation { instruction, call } = new_cx.pending_queue.pop();
          new_cx.current_call = call;
          new_s.instruction_counter = instruction;
        } else {
          // nothing pending (and nothing has returned yet either)
          // check whether there are still built-in functions that might return
          let event_loop = get_event_loop(new_cx)
          if event_loop.has_pending_calls() {
            event_loop.block_until_something_has_finished();
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

        // calling a built-in asynchronous function
        Value::AsyncBuiltIn(a) => {
          // execution fails if the wrong number of arguments is supplied
          if A::arity(a) != args.count() {
            return Status::Nope;
          } else {
            // retrieve the state corresponding to this built-in function
            let s = s.built_in_states.get(b);

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
                    children: Map::empty(),
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

                let frame = StackFrame {
                  return_instruction: s.instruction_counter + 1,
                  dst,
                  scope,
                  values,
                };

                let new_cx = cx;
                new_cx.stack = cx.stack.push(frame);
                new_s.async_stack[s.async_stack.count() - 1] = new_cx;
                new_s.instruction_counter = header;
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

// recursively clean up all children of the call that returned
// (remove itself and all children from pending_queue and result_queue)
fn clean_up_active_call(call_id: AsyncCallId, cx: AsyncContext) -> AsyncContext {
  let new_cx = cx;

  match cx.active_calls[call_id] {
    Instruction::BuiltIn(_) => {
      // cancel the pending call
      let event_loop = get_event_loop(cx);
      event_loop.cancel(call_id);

      // the call might have already returned
      // remove all matching return values from the return queue
      for (i, r) in new_cx.return_queue.enumerate() {
        if r.call == call_id {
          new_cx.return_queue = new_cx.return_queue.remove(i);
        }
      }

      return new_cx;
    }
    Regular {
      scope,
      values,
      continuation,
      children,
    } => {
      // remove all matching return values from the return queue
      for (i, r) in new_cx.return_queue.enumerate() {
        if r.call == call_id {
          new_cx.return_queue = new_cx.return_queue.remove(i);
        }
      }

      // remove all pending calls
      for (i, p) in new_cx.pending_queue.enumerate() {
        if p.call == call_id {
          new_cx.pending_queue = new_cx.pending_queue.remove(i);
        }
      }

      // recursively clean up children
      for child in children.keys() {
        new_cx = clean_up_active_call(child, new_cx);
      }

      return new_cx;
    }
  }
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
