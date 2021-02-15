function [output,interm_param] = Dense_forward(input,weight,bias,option)
% Dimension
% input: (1,features)
% weight{1,1}: (features,hidden_unit)
% bias{1,1}: (hidden_unit,1)
% output: (1,features)

interm_param = input*weight{1,1} + bias{1,1}';
if strcmp('sigmoid',option)
    output = sigmf(interm_param, [1 0]);
end
if strcmp('hyperbolic tangent',option)
    output = tanh(interm_param);
end
if strcmp('ReLU',option)
    output = max(interm_param,zeros(size(interm_param)));
end
if strcmp('softmax',option)
    output = softmax(interm_param');
    output = output';
end
end