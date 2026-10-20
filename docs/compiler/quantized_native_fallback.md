# Floating native families and the quantized contract

An exact native-dtype proof takes precedence. When it fails, a floating source
in a native arithmetic family is eligible under the declared quantized
computation. IEEE accumulation order, competing NaN word priority and reduced
storage conversion layout are not obligations of that computation. Integer
source semantics continue to require the native exact route. Movement and
alias policy remain separate.

The general rule in `numerical_semantics.quantized_native_fallback` is marked
**authored, pending operator review**. Its correctness criterion is precision
class (b): exact equality with the independent rational quantized reference.
Eager floating output is informational for that route.

The host quantizes each real input component with exact RNE and byte clamping.
The enclosing power-of-two scale comes from the declared M3 observer domain;
evaluation payloads never calibrate a scale, constrain component counts or
select a kernel. For the current range [-1,1], scale is 1/64 and the complete
code interval is [-128,127]. Every NaN sign/payload maps to storage code zero.
Negative and positive infinity map to the minimum and maximum code. Finite
rational body semantics do not license nonfinite-producing or bit-observing
operations.

Signed-byte component producers execute on the device, with full int32 readout.
Host recombination retains the exact numerator, exponent and divisor. The
exported result is quantized once by exact RNE/clamp, then dequantized to its
declared storage. Threshold parity is determined before any zero-point shift.
Representation splitting does not grant intermediate source rounding. Complete
provider coverage remains required; a partial prelude cannot place retained
host arithmetic.

Complex storage uses a final real/imaginary axis. Linear component families use
independent component quantization; runtime complex products require a
composition under strict native policy. Quantization does not grant native
status to runtime parallel products, nonlinear rounding constructions,
nonuniform result tables, index-generated payload arithmetic or unknown payload
carriers. A literal affine offset followed by native storage rounding instead
has its rounding at the quantized output boundary.

The independent oracle evaluates the original case arguments in rational
arithmetic, then applies the byte boundaries. Complex numbers are pairs of
rational numbers. It imports neither component lowering nor provider outputs.
An oracle failure is a qualification error and cannot choose host placement
based on payload data. Metadata, post-input storage, aliases and mutations
retain the existing separate whole-call checks. FireSim expectations come from
this oracle and independently recorded input storage.
