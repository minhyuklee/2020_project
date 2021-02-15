function [output_input, output_weight, output_bias] = stretch(input,weight,bias,stride)
% input, weight, bias를 matrix 꼴로 convolution하기 좋게 변환해주는 함수.
input_shape = size(input); % (time_step, feature, filter(channel))
kernel_size = size(weight{1,1},1);
num_filter = size(bias,1);
output_length = (input_shape(1)-kernel_size)/stride + 1; % stride는 미리 딱 떨어지게 계산되어 정해졌다고 가정.

if length(input_shape) == 2 % input이 1d 또는 2d인 경우
    output_input = zeros(kernel_size*input_shape(2),output_length);
    output_weight = zeros(num_filter,kernel_size*input_shape(2));
    output_bias = zeros(num_filter,output_length);

    % output_input 얻는 과정
    for i=1:output_length
        tmp = [];
        for j=1:kernel_size
            tmp = [tmp;input(j+stride*(i-1),:)'];
        end
        output_input(:,i) = tmp;
    end
    
    % output_weight, output_bias 얻는 과정
    for i=1:num_filter
        tmp = [];
        for j=1:kernel_size
            tmp = [tmp,weight{i,1}(j,:)];
        end
        output_weight(i,:) = tmp;
        output_bias(i,:) = bias{i,1}*ones(1,output_length);
    end
end

if length(input_shape) == 3 % input이 3d인 경우
    output_input = zeros(kernel_size*input_shape(2)*input_shape(3),output_length);
    output_weight = zeros(num_filter,kernel_size*input_shape(2)*input_shape(3));
    output_bias = zeros(num_filter,output_length);
    
    % output_input 얻는 과정
    for i=1:output_length
        tmp = [];
        for j=1:input_shape(3)
            for k=1:kernel_size
                tmp = [tmp,input(k+stride*(i-1),:,j)'];
            end
        end
        output_input(:,i) = tmp;
    end
    
    % output_weight, output_bias 얻는 과정
    for i=1:num_filter
        tmp = [];
        for j=1:input_shape(3)
            for k=1:kernel_size
                tmp = [tmp,weight{i,1}(k,:,j)];
            end
        end
        output_weight(i,:) = tmp;
        output_bias(i,:) = bias{i,1}*ones(1,output_length);
    end
end
    
end