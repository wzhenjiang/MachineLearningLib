function [normalized_data, mu, sigma] = mla_mean_normalization(data, mu = [], sigma = [], stderr = false)
%% Purpose:	This function is to normalize features to a certain range
%% Purpose: The range is [.]
%% Info: mean is used for mu, range/std would be used for sigma per stderr para
%% Info: if range is used, output is between [-0.5, 0.5]
%% Info: if std is used, 99.7% data would be in [-3.3]
%% Attention: Your application need to save mu and sigma for later normalization of your input X
%% Attention: If mu and sigma is presented, normalization would be according to given value

% initiate variable
normalized_data = [];

% prepare mu and sigma (1 * n matrix)
if size(mu,1) == 0  || size(sigma,1) == 0
	mu = mean(data);
	if stderr
		sigma = std(data);
	else
		sigma = range(data);
end;

% form normalized_data
[m,n] = size ( data )

for i = 1: m
	normalized_data = [normalized_data; (data(i,:) - mu ) ./ sigma];
end;

end