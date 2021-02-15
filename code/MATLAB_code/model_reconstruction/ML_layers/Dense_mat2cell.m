function [weight_Dense, bias_Dense] =Dense_mat2cell(weight,bias)
    weight_Dense = cell(1,1);
    weight_Dense{1,1} = weight;
    bias_Dense = cell(1,1);
    bias_Dense{1,1} = bias;
end
