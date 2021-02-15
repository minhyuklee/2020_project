function output = Flatten(input)
dim_check = size(input);

if length(dim_check) == 3 % input이 3D인 경우
    output = zeros(1,dim_check(1)*dim_check(2)*dim_check(3));
    for i=1:dim_check(1)
        tmp = input(i,:,:);
        output(1,dim_check(3)*(i-1)+1:dim_check(3)*i) = reshape(tmp,[1,dim_check(3)]);
    end
%     for i=1:dim_check(3)
%         for j=1:dim_check(1)
%             output(1,dim_check(1)*dim_check(2)*(i-1)+dim_check(2)*(j-1)+1:dim_check(1)*dim_check(2)*(i-1)+dim_check(2)*j) = input(j,:,i);
%         end
%     end

end