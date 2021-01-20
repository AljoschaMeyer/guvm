# Generic Unityped Virtual Machine

The generic unityped virtual machine (guvm) is a simple virtual machine for unityped (i.e. dynamically typed) programming languages whose definition is generic over the exact type of values. The language might support integers, arrays, whatevers, the guvm doesn't care. What it does provide are first-class function values (lexically scoped closures).

Functions work by executing simple instructions such as loading/storing values from/to memory, (conditionally) jumping to other instructions, or calling further functions. The semantics are kept fairly simple, there is no exception handling and all functions are of fixed arity.

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

`Map<A, B>` is the type of partial mappings from instances of type `A` to instances of type `B`. Let `m: Map<A, B>`, `a: A`, `b: B`. Then `m.get(a): B` denotes the value to which `m` maps `a` (and is undefined if `m` does not map `a`). `m.insert(a, b): Map<A, B>` denotes an updated mapping that works just like `m` except that it maps `a` to `b`.

`[A]` this is the type of finite sequences containing `A`s. Let `s: [A]`, `a: A`, `i: Nat`. Then `s.count(): Nat` denotes the number of elements in `s`, `s.push(a): [A]` denotes the sequence obtained by adding `a` as the new last element of `s`, `s.pop(): [A]` denotes the sequence obtained by removing the last element of `s` (undefined if `s` is the empty sequence) `s[i]` denotes the i-th element within `s` (0-based indexing, undefined if the sequence is too short), `s.last(): [A]` denotes `s[s.count() - 1]`, `[]` denotes the empty sequence.

We finally define some type synonyms that will help clarifying what certain index values are intended to be used for:

```rust
type ScopeId = Nat;
type InstructionIndex = U64;
type GlobalIndex = U64;
type LocalIndex = U64;
type ScopeIndex = U64;
```

Whenever the definition of the virtual machine involves natural numbers, an implementation may substitute a finite subset of the natural numbers.

The specification of the language semantics involves expressions that may be undefined, e.g. `m.get(a)` for a mapping `m` that does not map `a`. The specification assumes that prior to running the virtual machine, a conservative check has been performed that makes sure that undefined cases never arise. Since the virtual machine is fairly static (no arbitrarily-computed jumps, fixed numbers of arguments, stack space and heap space for functions), such a check can be performed in linear time. Alternatively, an implementation can forgo such checks, instead exhibiting undefined behavior in any such case. This is not recommended when running untrusted code, but it makes sense when the code is known to be produced from a trusted source, e.g. a compiler that only emits a valid code.

## Semantics

To define the virtual machine semantics, we need to introduce three different concepts: the set of runtime `Value`s, the `Instruction`s that are executed by the vm, and the `Context` in which instructions executed.

### Values

A `Value` is a runtime value of the vm:

```rust
enum Value {
  v: V,
  b: B,
  f: Function, // to be defined later
}
```

`V` can be chosen almost arbitrarily, the only constraint is that there is a function `V::truthy(v: V) -> Bool` that assigns to each `V` whether it triggers conditional jump instructions are not.

`B` can also be chosen mostly arbitrarily, there are two constraints: there has to be a function `B::arity(b: B) -> Nat` specifying how many arguments the built-in function expects, and `B::invoke(b: B, state: S, arguments: [Value]) -> Option<(S, Value)>`, which fallibly computes a return value and updates the functions state. The precise workings are explained later when the semantics of the call instruction are given.

```rust
struct Function {
  // unique identifier for this value
  ordinal: Nat,
  // access to closed-over values
  env: EnvironmentIndex,
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
  // which environment can be accessed from the current function call
  environment: EnvironmentIndex,
  // values that can be accessed only from within the current function call
  local: [Value],
}
```

### Instructions
