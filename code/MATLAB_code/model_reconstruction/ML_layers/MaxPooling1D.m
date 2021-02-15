function output = MaxPooling1D(input,pooling_size)
% input = (time_step,feature,channel)
% time_step을 pooling_size의 크기로 등분하여 각 구간 내 최대값을 추출하여 output을 재구성하는 작업.
dim_check = size(input);

if length(dim_check) == 2 % input이 1D, 2D인 경우
    output = zeros(dim_check(1)/pooling_size, 1);
    for i=1:size(output,2)
        output(1,i) = max(max(input(:,pooling_size*(i-1)+1:pooling_size*i)));
    end
end

if length(dim_check) == 3 % input이 3D인 경우
    output = zeros(dim_check(1)/pooling_size, 1, dim_check(3));
    for i=1:size(output,3)
        for j=1:size(output,1)
            output(j,1,i) = max(max(input(pooling_size*(j-1)+1:pooling_size*j,:,i)));
        end
    end
end

end