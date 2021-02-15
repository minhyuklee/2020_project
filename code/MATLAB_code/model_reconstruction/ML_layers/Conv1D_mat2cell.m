function [weight_Conv1D, bias_Conv1D] = Conv1D_mat2cell(weight,bias,kernel_size)
num_filter = length(bias); % = length(weight)/kernel_size
weight_Conv1D = cell(num_filter,1);
bias_Conv1D = cell(num_filter,1);
% weight_Conv1D의 각 cell element에는 kernel이 포함되어 있음.
for i=1:size(weight_Conv1D,1)
    weight_Conv1D{i,1} = weight(kernel_size*(i-1)+1:kernel_size*i,:);
    bias_Conv1D{i,1} = bias(i,1);
end

end