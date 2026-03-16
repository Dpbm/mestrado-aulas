
function convolve(a,b)
  results = [];
  
  inc_sides = length(a) - 1;
  total_size = length(b) + 2*inc_sides;

  ext_a = vcat(a, zeros(length(b) + inc_sides - 1));
  ext_b = vcat(zeros(inc_sides), b, zeros(inc_sides));
  
  total_iter = length(a) + length(b) - 1;
  
  for _ in 1:total_iter-1
    value = 0;
    for (i,v) in pairs(ext_a)
      value += v*ext_b[i];
    end
    push!(results,value);
    deleteat!(ext_a, total_size);
    pushfirst!(ext_a, 0);
  end

  results

end
