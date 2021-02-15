function output = repetition_removal(input)

pc = pca(input');
pc1 = pc(:,1);
prj = [];
for k=1:size(input,2)
    prj = [prj,dot(input(:,k),pc1)];
end
input_1d = prj;

% local maxima 찾기
[~,locs_max] = findpeaks(input_1d);

% x = 0.008*(1:1:size(input_1d,2));
% figure(1)
% plot(x,input_1d,x(locs_max),input_1d(locs_max),'r*','MarkerSize',6)
% grid on
% set(gca,'FontSize',16)
% ylabel('1st projection')
% xlabel('time step')

% local minima 찾기
minima_bool = islocalmin(input_1d);
org_ind = 1:1:size(input_1d,2);
locs_min = org_ind(minima_bool);

% x = 0.008*(1:1:size(input_1d,2));
% figure(2)
% plot(x,input_1d,x(locs_min),input_1d(locs_min),'r*','MarkerSize',6)
% grid on
% set(gca,'FontSize',16)
% ylabel('1st projection')
% xlabel('time step')

important_locs = [1,length(input_1d)];

tol = 0.15*abs(max(input_1d) - min(input_1d));

first_lower_tol_temp = find(abs(input_1d - input_1d(1)) < tol);
first_lower_tol = first_lower_tol_temp(1);
for i=2:length(first_lower_tol_temp) % 인덱스의 연속성 판단
    if first_lower_tol_temp(i) - first_lower_tol_temp(i-1) == 1
        first_lower_tol = [first_lower_tol,first_lower_tol_temp(i)];
    else
        break
    end
end
important_locs = [important_locs, first_lower_tol];

last_lower_tol_temp = find(abs(input_1d - input_1d(end)) < tol);
last_lower_tol = last_lower_tol_temp(end);
for i=length(last_lower_tol_temp)-1:-1:1 % 인덱스의 연속성 판단
    if last_lower_tol_temp(i+1) - last_lower_tol_temp(i) == 1
        last_lower_tol = [last_lower_tol,last_lower_tol_temp(i)];
    else
        break
    end
end
important_locs = [important_locs, last_lower_tol];

important_locs = sort([important_locs,locs_max,locs_min]);    
    
candidate_first_peak = find(abs(input_1d(important_locs) - input_1d(1)) > tol);
candidate_last_peak = find(abs(input_1d(important_locs) - input_1d(end)) > tol);

first_peak = important_locs(candidate_first_peak(1));
last_peak = important_locs(candidate_last_peak(end));

% x = 0.008*(1:1:size(input_1d,2));
% figure(3)
% plot(x,input_1d,'LineWidth',1.5)
% hold on
% plot([x(first_peak),x(first_peak)],[min(input_1d),max(input_1d)],'r','LineWidth',1.5)
% hold on
% plot([x(last_peak),x(last_peak)],[min(input_1d),max(input_1d)],'r','LineWidth',1.5)
% grid on
% set(gca,'FontSize',16)
% xlim([x(1),x(end)])
% ylabel('1st projection')
% xlabel('time step')

for p1 = last_peak+1:size(input_1d,2)
    difference_before = abs(input_1d(first_peak) - input_1d(p1-1));
    difference_after = abs(input_1d(first_peak) - input_1d(p1));
    if difference_after >= difference_before
        break
    end
end
for p2 = first_peak-1:-1:1
    difference_before = abs(input_1d(last_peak) - input_1d(p2+1));
    difference_after = abs(input_1d(last_peak) - input_1d(p2));
    if difference_after >= difference_before
        break
    end
end

% x = 0.008*(1:1:size(input_1d,2));
% figure(4)
% plot(x,input_1d)
% hold on
% plot([x(first_peak),x(first_peak)],[min(input_1d),max(input_1d)],'r')
% hold on
% plot([x(last_peak),x(last_peak)],[min(input_1d),max(input_1d)],'r')
% hold on
% plot([x(p1),x(p1)],[input_1d(p1),input_1d(p1)],'o','Color','r','MarkerSize',10)
% hold on
% plot([x(p2),x(p2)],[input_1d(p2),input_1d(p2)],'o','Color','g','MarkerSize',10)
% grid on
% set(gca,'FontSize',16)
% ylabel('1st projection')
% xlabel('time step')
% 
% figure(5)
% x_1 = 1:1:first_peak;
% x_2 = last_peak:1:length(input_1d);
% plot(x_1,input_1d(1:first_peak),'b',x_2,input_1d(last_peak:end),'b')
% hold on
% plot(first_peak,input_1d(first_peak),'o','Color','k','MarkerSize',10)
% hold on
% plot(last_peak,input_1d(last_peak),'o','Color','k','MarkerSize',10)
% hold on
% plot(p1,input_1d(p1),'o','Color','k','MarkerSize',10)
% hold on
% plot(p2,input_1d(p2),'o','Color','k','MarkerSize',10)
% grid on
% set(gca,'FontSize',16)
% ylabel('1st projection')

seq1 = input_1d(p2:first_peak);
seq2 = input_1d(last_peak:p1);
[~,opt_index_sq1,opt_index_sq2] = dtw(seq1,seq2);
opt_index_org_sq1 = opt_index_sq1 + p2 - 1;
opt_index_org_sq2 = opt_index_sq2 + last_peak - 1;
mu = linspace(0,1,length(opt_index_sq1));
mu_mat = repmat(mu,10,1);
optimal_connection = mu_mat.*input(:,opt_index_org_sq2) + (ones(size(mu_mat)) - mu_mat).*input(:,opt_index_org_sq1);
output = [input(:,1:p2-1),optimal_connection,input(:,p1+1:end)];

end