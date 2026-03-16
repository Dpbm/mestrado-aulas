include("conv.jl");

f(t) = cos(4*pi*t) + cos(10*pi*t);
samples = 1:12;
result = f.(samples);

g = [1/sqrt(2), -1/sqrt(2)];

print(convolve(result,g));


