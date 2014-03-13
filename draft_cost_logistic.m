function mla_compute_cost_logistic()
%% Purpose: this function is to compute cost of logistic function

cost = -y * log(h) - (1-y) * log(1-h);
end
