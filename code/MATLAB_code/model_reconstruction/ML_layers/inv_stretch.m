function output = inv_stretch(input)
% matrix로 변환한 형태로 convolution 취한 결과를 다시 원래 형태로 돌려놓기 위한 함수.
% input = (num_filter,output_length), stretch 함수 참고
output = zeros(1,size(input,2),size(input,1));

for i=1:size(input,1)
    output(:,:,i) = input(i,:);
end

end