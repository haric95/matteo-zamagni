type Arguments<F extends Function> = F extends (...args: infer A) => any
  ? A
  : never;
