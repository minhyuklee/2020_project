function [output, final_index] = seq_compress_v2(data,tol)
% 패턴 내에서 편차가 가장 큰 센서 찾기
max_dev_temp = [];
for i=1:size(data,1)
    max_dev_temp = [max_dev_temp,max(data(i,:))-min(data(i,:))];
end
[~,max_dev_sensor] = max(max_dev_temp);

P = data(max_dev_sensor,:); % 가장 편차가 큰 센서의 패턴 선정
P_start = 1;
P_end = size(P,2);
candidate = [];
while 1 % 몇 단계에 걸쳐 simplification 할 것인지에 대한 루프
    candsize_old = size(candidate,2);
   indexing = [P_start,candidate,P_end];
   for i = 1:size(indexing,2)-1 % 각 단계에서 candidate 구하는 것에 대한 루프
       segment_lower = [indexing(1,i);P(indexing(1,i))]; % [x;y]
       segment_upper = [indexing(1,i+1);P(indexing(1,i+1))]; % [x;y]
       slope = (segment_upper(2)-segment_lower(2))/(segment_upper(1)-segment_lower(1));
       bias = -segment_lower(1)*slope + segment_lower(2);
       
       if segment_upper(1) - segment_lower(1) == 1
           continue;
       end
       distance_tmp = []; % [거리;위치]
       for j=segment_lower(1)+1:segment_upper(1)-1
           distance_tmp = [distance_tmp,[abs(slope*j-P(1,j)+bias)/sqrt(slope^2+1);j]];
       end
       [max_dist,max_dist_ind] = max(distance_tmp(1,:));
       if max_dist >= tol
           candidate = [candidate,distance_tmp(2,max_dist_ind)];
       end
       candidate = sort(candidate);
   end
   candsize = size(candidate,2);
   if candsize - candsize_old == 0
       break;
   end
end
final_index = [P_start,candidate,P_end];
output = data(:,final_index);
end