function [weight_LSTM,bias_LSTM] = LSTM_mat2cell(weight_x,weight_h,bias,units)
% keras.model에 LSTM layer가 있을 경우 사용.
% keras.model은 LSTM layer의 weight와 bias를 추출 시 각 gate의 weight/bias를 가로 방향으로
% 연달아 이어서 하나의 matrix 형태로 만들어줌.
% MATLAB 코드로 만들어진 LSTM forward pass 함수는 input으로 weight와 bias를 받는데, 그 형태는
% 다음과 같음.

% weight cell
% (1,1): W_xf, (1,2): W_hf
% (2,1): W_xi, (2,2): W_hi
% (3,1): W_xc, (3,2): W_hc
% (4,1): W_xo, (4,2): W_ho

% bias cell
% (1,1): bias_f
% (2,1): bias_i
% (3,1): bias_c
% (4,1): bias_o

% 따라서 각 gate(input, forget, state, output gate) 별로 구분하여 weight/bias cell을 만들어주어야 함.
x_i = weight_x(:,1:units);
x_f = weight_x(:,units+1:2*units);
x_c = weight_x(:,2*units+1:3*units);
x_o = weight_x(:,3*units+1:4*units);

h_i = weight_h(:,1:units);
h_f = weight_h(:,units+1:2*units);
h_c = weight_h(:,2*units+1:3*units);
h_o = weight_h(:,3*units+1:4*units);

b_i = bias(1:units,:);
b_f = bias(units+1:2*units,:);
b_c = bias(2*units+1:3*units,:);
b_o = bias(3*units+1:4*units,:);

weight_LSTM = cell(4,2);
weight_LSTM{1,1} = x_f; weight_LSTM{1,2} = h_f;
weight_LSTM{2,1} = x_i; weight_LSTM{2,2} = h_i;
weight_LSTM{3,1} = x_c; weight_LSTM{3,2} = h_c;
weight_LSTM{4,1} = x_o; weight_LSTM{4,2} = h_o;

bias_LSTM = cell(4,1);
bias_LSTM{1,1} = b_f;
bias_LSTM{2,1} = b_i;
bias_LSTM{3,1} = b_c;
bias_LSTM{4,1} = b_o;
end