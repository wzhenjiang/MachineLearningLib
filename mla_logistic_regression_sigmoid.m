function h = mla_classification_sigmoid(z)
%% Purpose: given z, compute g(z)

h = 1 ./ ( 1 + e ^ (-z) );

end
