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

Another precondition for the semantics to make sense is that the ordinals of all functions in a given `Context` are distinct.

## Semantics

To define the virtual machine semantics, we need to introduce three different concepts: the set of runtime `Value`s, the `Instruction`s that are executed by the vm, and the `Context` in which instructions executed.

### Values

A `Value` is a runtime value of the vm:

```rust
enum Value {
  V(V),
  BuiltIn(B),
  Function(Function), // to be defined later
}
```

`V` can be chosen almost arbitrarily, the only constraints are that there is a function `V::truthy(v: V) -> Bool` that assigns to each `V` whether it triggers conditional jump instructions or not, and that there is a function `V::default() -> V`.

`B` can also be chosen mostly arbitrarily, there are two constraints: there has to be a function `B::arity(b: B) -> Arity` specifying how many arguments the built-in function expects, and `B::invoke(b: B, state: S, arguments: [Value]) -> Option<(S, Value)>`, which fallibly computes a return value and updates the functions state. The precise workings are explained later when the semantics of the call instruction are given.

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

### Context

A `Context` describes the full state of the virtual machine at a single point in time. The machine works by iteratively computing a new `Context` from the previous one until a termination condition has been reached.

```rust
// the virtual machine state
enum Context {
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
// specifies how to compute the next context from the current one
enum Instruction {
  // jump to a specific instruction
  Jump(InstructionIndex),
  // jump to a specific instruction if the address holds a truthy value
  ConditionalJump { condition: Address, target: InstructionIndex },
  // copy the value from the first address to the second address
  Assign { src: Address, dst: Address },
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
fn load(src: Address, c: Context) -> Value {
  match src {
    Address::Global(i) => c.global_map.get(i),
    Address::Local(i) => c.stack.last().values[i],
    Address::Scoped { up, index } => load_scoped(c.stack.last().scope, c, up, index),
  }
}

fn load_scoped(
  scope: ScopeId,
  c: Context,
  up: AncestorDistance,
  i: ScopeIndex,
) -> Value {
  if up == 0 {
    c.scope_map[scope].values[i]
  } else {
    match c.scope_map[scope].parent {
      Some(parent) => load_scoped(parent, c, up - 1, i),
      None => undefined!(),
    }
  }
}
```

The semantics of storing a value at an address:

```rust
fn store(v: Value, dst: Address, c: Context) -> Context {
  let mut new_c = c;
  match dst {
    Address::Global(i) => new_c.global_map.insert(i, v),
    Address::Local(i) => new_c.stack.last().values[i] = v,
    Address::Scoped { up, index } => new_c = store_scoped(v, c.stack.last().scope, c, up, index),
  };
  return new_c;
}

fn store_scoped(
  v: Value,
  scope: ScopeId,
  c: Context,
  up: AncestorDistance,
  i: ScopeIndex,
) -> Context {
  if up == 0 {
    let mut new_c = c;
    new_c.scope_map[scope].values[i] = v;
    return new_c;
  } else {
    match c.scope_map[scope].parent {
      Some(parent) => store_scoped(v, parent, c, up - 1, i),
      None => undefined!(),
    }
  }
}
```

With these operations defined, we can now specify the semantics of the virtual machine. The virtual machine operates by continuously computing a new `Context` from an old one until it either successfully yields a `Value` or cannot continue.

```rust
// trying to go from one context to the next can have three different outcomes
enum Status {
  // there is no successor context because the computation finished
  Done(Value),
  // there is no successor context because something went wrong
  Nope,
  // computed the successor context
  Continue(Context),
}

// compute the successor context by performing a single instruction
fn step(c: Context) -> Status {
  // the successor context is a modified version of the old one
  let mut new_c = c;

  // perform the instruction indicated by the current instruction counter
  match c.instruction_map.get(c.instruction_counter) {
    // unconditional jump
    Instruction::Jump(target) => new_c.instruction_counter = target,

    // jump if the value at the condition address is truthy
    Instruction::ConditionalJump { condition, target } => {
      if V::truthy(load(condition, c)) {
        new_c.instruction_counter = target;
      } else {
        new_c.instruction_counter += 1;
      }
    }

    // load src and store it to dst
    Instruction::Assign { src, dst } => {
      new_c = store(load(src, c), dst, c);
      new_c.instruction_counter += 1;
    }

    // function headers are ignored during execution, they merely store data
    Instruction::FunctionHeader { arity, local, scoped } => new_c.instruction_counter += 1,

    // calling functions is the most complicated instruction
    Instruction::Call { dst, callee, arguments } => {
      // resolve the value to call
      let f = load(callee, c);

      // resolve the arguments
      let mut args = [];
      for (i, argument) in arguments.enumerate() {
        args = args.push(load(argument, c));
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
            let s = c.built_in_states.get(b);

            // delegate to the implementation of the built-in function
            match B::invoke(b, s, args) {
              // the implementation decided to halt the computation
              None => Status::Nope,
              // successfully computed a value and a new state for the function
              Some((new_s, v)) => {
                new_c = store(v, dst, c);
                new_c.built_in_states = c.built_in_states.insert(b, new_s);
                new_c.instruction_counter += 1;
              }
            }
          }
        }

        // calling a regular function
        Value::Function(Function { ordinal, scope, header }) => {
          // look up information from the function header
          match c.instruction_map.get(header) {
            Instruction::FunctionHeader { arity, local, scoped } => {
              // execution fails if the wrong number of arguments supplied
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
                  return_instruction: c.instruction_counter + 1,
                  dst,
                  scope,
                  values,
                };

                new_c.stack = c.stack.push(frame);
                new_c.instruction_counter = header;
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
      match c.instruction_map.get(header) {
        Instruction::FunctionHeader { arity, local, scoped } => {
          // choose an ordinal for the new function value
          let ordinal = magically_conjure_previously_unused_ordinal();

          // create a new scope for the function value
          let scope_id = c.scope_map.fresh_key();
          let mut values = [];
          for _ in 0..scoped {
            values = values.push(V::default());
          }
          let new_scope = Scope {
            parent: Some(c.stack.last().scope),
            values,
          }

          let f = Function {
            ordinal,
            scope: new_scope,
            header,
          };

          // store new function value to dst
          new_c = store(f, dst, c);

          // add the new scope to the context
          new_c.scope_map = c.scope_map.insert(scope_id, new_scope);

          new_c.instruction_counter += 1;
        }

        // header must point to a function header instruction
        _ => undefined!(),
      }
    }

    // end a function call, having computed the value stored at src
    Instruction::Return(src) => {
      // the returned value
      let v = load(src, c);

      // if there is no more call stack left, the vm has computed a value
      if c.stack.count() == 1 {
        return Status::Done(v);
      } else {
        // find out where to continue execution
        let StackFrame {
          return_instruction,
          dst,
          ..
        } = c.stack.last();

        // pop from the stack
        new_c.stack = c.stack.pop();

        // store the function result
        // note that the dst step is resolved after popping
        new_c = store(v, dst, new_c);

        // continue execution from where the function was called
        new_c.instruction_counter = return_instruction;
      }
    }
  }

  return Status::Continue(new_c);
}

// the semantics of a virtual machine with initial state `c`
fn run(mut c: Context) -> Option<Value> {
  while true {
    match step(c) {
      Status::Done(v) => return Some(v),
      Status::Nope => return None,
      Status::Continue(new_c) => c = new_c,
    }
  }
}
```

A note on determinism: `magically_conjure_previously_unused_ordinal(): Nat` is the only nondeterministic part of the specification. Practical implementations of dynamically typed languages often identify functions by some heap address, which is provided nondeterministically from an allocator. To obtain deterministic semantics, the virtual machine can keep a counter, incrementing it by one and then using the resulting number whenever a new ordinal is required. This version of the virtual machine is called the **deterministic generic unityped virtual machine** (dguvm).

## Asynchronous (Deterministic) Generic Unityped Virtual Machine

TODO
