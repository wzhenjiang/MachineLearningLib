function [h, output_matrix] = test_plot_neuon_network_sample_data(data, width)
%% Purpose: this function is to plot the sample data in a colormap

% if width is not provided, compute it. 
if ~exist('width', 'var') || isempty(width) 
	width = round( sqrt( size(data, 2) ) );	% assume the data is square matrix
end;

% Compute rows, cols
[m n] = size(data);
height = (n / width);

% Compute grid
rows = floor(sqrt(m));
cols = ceil(m / rows);

% set pad between images
pad = 1;

% generate display array and set to -1
output_matrix = - ones( rows * (height + pad) + pad, cols * (width + pad) + pad);

for i=1:m
	row = ceil(i / cols);
	col = mod(i - 1, cols) + 1;
	image_data = reshape(data(i,:),height,width);
	scaled_image_data = image_data / max(data(i,:));
	output_matrix( (row-1) * (height+pad) + pad + ( 1: height ), ...
					(col-1) * (width + pad) + pad + ( 1: width) ) = ...
					scaled_image_data;
end;

% Display Image
colormap(gray);
h = imagesc(output_matrix, [-1 1]);
axis image off
drawnow;

end
