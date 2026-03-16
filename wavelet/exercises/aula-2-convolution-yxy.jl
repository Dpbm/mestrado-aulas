# Y[.] * Y[.] = Y'[.]
#
include("conv.jl");

y = [24,34,56,72,46,20];

print(convolve(y,y));
