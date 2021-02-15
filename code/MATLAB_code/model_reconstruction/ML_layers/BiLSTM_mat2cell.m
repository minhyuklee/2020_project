function [weight_LSTM,bias_LSTM] = BiLSTM_mat2cell(weight_x_forward,weight_h_forward,bias_forward,weight_x_backward,weight_h_backward,bias_backward,units)
% keras.model에 Bidirectional-LSTM layer가 있을 경우 사용.
% keras.model은 Bi-LSTM layer의 weight와 bias를 추출 시 각 gate의 weight/bias를 가로 방향으로
% 연달아 이어서 하나의 matrix 형태로 만들어줌.
% MATLAB 코드로 만들어진 Bi-LSTM forward pass 함수는 input으로 weight와 bias를 받는데, 그 형태는
% 다음과 같음.

% weight cell
% (1,1): W_xf_forward, (1,2): W_hf_forward
% (2,1): W_xi_forward, (2,2): W_hi_forward
% (3,1): W_xc_forward, (3,2): W_hc_forward
% (4,1): W_xo_forward, (4,2): W_ho_forward
% (5,1): W_xf_backward, (5,2): W_hf_backward
% (6,1): W_xi_backward, (6,2): W_hi_backward
% (7,1): W_xc_backward, (7,2): W_hc_backward
% (8,1): W_xo_backward, (8,2): W_ho_backward

% bias cell
% (1,1): bias_f_forward
% (2,1): bias_i_forward
% (3,1): bias_c_forward
% (4,1): bias_o_forward
% (5,1): bias_f_backward
% (6,1): bias_i_backward
% (7,1): bias_c_backward
% (8,1): bias_o_backward

% 따라서 각 gate(input, forget, state, output gate) 별로 구분하여 weight/bias cell을 만들어주어야 함.

% forward direction
x_i_forward = weight_x_forward(:,1:units);
x_f_forward = weight_x_forward(:,units+1:2*units);
x_c_forward = weight_x_forward(:,2*units+1:3*units);
x_o_forward = weight_x_forward(:,3*units+1:4*units);

h_i_forward = weight_h_forward(:,1:units);
h_f_forward = weight_h_forward(:,units+1:2*units);
h_c_forward = weight_h_forward(:,2*units+1:3*units);
h_o_forward = weight_h_forward(:,3*units+1:4*units);

b_i_forward = bias_forward(1:units,:);
b_f_forward = bias_forward(units+1:2*units,:);
b_c_forward = bias_forward(2*units+1:3*units,:);
b_o_forward = bias_forward(3*units+1:4*units,:);

% backward direction
x_i_backward = weight_x_backward(:,1:units);
x_f_backward = weight_x_backward(:,units+1:2*units);
x_c_backward = weight_x_backward(:,2*units+1:3*units);
x_o_backward = weight_x_backward(:,3*units+1:4*units);

h_i_backward = weight_h_backward(:,1:units);
h_f_backward = weight_h_backward(:,units+1:2*units);
h_c_backward = weight_h_backward(:,2*units+1:3*units);
h_o_backward = weight_h_backward(:,3*units+1:4*units);

b_i_backward = bias_backward(1:units,:);
b_f_backward = bias_backward(units+1:2*units,:);
b_c_backward = bias_backward(2*units+1:3*units,:);
b_o_backward = bias_backward(3*units+1:4*units,:);

% combine
weight_LSTM = cell(8,2);
weight_LSTM{1,1} = x_f_forward; weight_LSTM{1,2} = h_f_forward;
weight_LSTM{2,1} = x_i_forward; weight_LSTM{2,2} = h_i_forward;
weight_LSTM{3,1} = x_c_forward; weight_LSTM{3,2} = h_c_forward;
weight_LSTM{4,1} = x_o_forward; weight_LSTM{4,2} = h_o_forward;
weight_LSTM{5,1} = x_f_backward; weight_LSTM{5,2} = h_f_backward;
weight_LSTM{6,1} = x_i_backward; weight_LSTM{6,2} = h_i_backward;
weight_LSTM{7,1} = x_c_backward; weight_LSTM{7,2} = h_c_backward;
weight_LSTM{8,1} = x_o_backward; weight_LSTM{8,2} = h_o_backward;

bias_LSTM = cell(8,1);
bias_LSTM{1,1} = b_f_forward;
bias_LSTM{2,1} = b_i_forward;
bias_LSTM{3,1} = b_c_forward;
bias_LSTM{4,1} = b_o_forward;
bias_LSTM{5,1} = b_f_backward;
bias_LSTM{6,1} = b_i_backward;
bias_LSTM{7,1} = b_c_backward;
bias_LSTM{8,1} = b_o_backward;
end