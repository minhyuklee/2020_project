function [output,stat_MS] = normalize(data)
tmp = [];
for i=1:size(data,3)
    tmp = cell2mat(data(:,:,i));
%     for j=1:size(data(:,1,i),1)
%         tmp = [tmp;data{j,1,i}];
%     end
end
stat_MS(1,:) = mean(tmp);
stat_MS(2,:) = std(tmp,0);
% stat_MS: 센서 별 mean과 std를 저장해놓은 변수
 
for i=1:size(data,3)
    for j=1:size(data(:,1,i),1)
        stat_MS_mean_mat = repmat(stat_MS(1,:),size(data{j,1,i},1),1);
        stat_MS_std_mat = repmat(stat_MS(2,:),size(data{j,1,i},1),1);
        data{j,1,i} = (data{j,1,i} - stat_MS_mean_mat)./stat_MS_std_mat;
%         for k=1:size(data{j,1,i},1)
%             data{j,1,i}(:,k) = (data{j,1,i}(:,k)-stat_MS(:,1))./stat_MS(:,2);
%         end
    end
end
output = data;
end