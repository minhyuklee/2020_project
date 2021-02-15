function hidden_set = BiLSTM_forward(input,weight,bias,return_sequences,merge_mode)
% Dimension
% input: (timestep,features)
% weight{:,1}: (features, hidden_unit)
% weight{:,2}: (hidden_unit, hidden_unit)
% bias{:,:}: (hidden_unit, 1)
% hidden_set: (timestep,hidden_unit)

weight_forward = weight(1:4,:);
weight_backward = weight(5:8,:);

bias_forward = bias(1:4,:);
bias_backward = bias(5:8,:);

input_backward = flipud(input);

[hidden_forward,~] = LSTM_forward(input,weight_forward,bias_forward);
[hidden_backward,~] = LSTM_forward(input_backward,weight_backward,bias_backward);
if strcmp('True', return_sequences)
    h_forward = hidden_forward;
    h_backward = flipud(hidden_backward);
elseif strcmp('False', return_sequences)
    h_forward = hidden_forward(end,:);
    h_backward = hidden_backward(end,:);
end
    
if strcmp('concat', merge_mode)
    hidden_set = [h_forward, h_backward];

elseif strcmp('sum', merge_mode)
    hidden_set = h_forward + h_backward;
    
elseif strcmp('ave', merge_mode)
    hidden_set = (h_forward + h_backward)/2;
    
elseif strcmp('mul', merge_mode)
    hidden_set = h_forward.*h_backward;
end

end