function [output, GPS_gesture] = GPS_trainset_generation_v2(input,window_size,sliding)
% input: cell(gesture sample,1,gesture index)
% output: cell(GPS training data, GPS)

% Step 1: PCA, 1st pc projection
principal = cell(size(input));
for i=1:size(input,3)
    for j=1:size(input,1)
        g_sample = input{j,1,i};

        pc = pca(g_sample');
        pc1 = pc(:,1);
        pc1_mag = norm(pc1);
        projection = [];
        for k=1:size(g_sample,2)
            projection = [projection,dot(g_sample(:,k),pc1)/pc1_mag];
        end
        principal{j,1,i} = projection;
    end
end
% figure()
% x_time = 0.008*(1:1:length(projection));
% plot(x_time,projection)
% grid on
% xlabel('time (sec)')
% ylabel('1st pc projection')
% set(gca,'FontSize',20)

% Step 2: Velocity calculation & taking absolute
velocity = cell(size(input));
for i=1:size(principal,3)
    for j=1:size(principal,1)
        vel_tmp = 0;
        for k=1:size(principal{j,1,i},2)-1
            vel_tmp = [vel_tmp, principal{j,1,i}(1,k+1) - principal{j,1,i}(1,k)];
        end
        velocity{j,1,i} = abs(vel_tmp);
    end
end
% figure()
% x_time = 0.008*(1:1:length(vel_tmp));
% plot(x_time,vel_tmp)
% grid on
% xlabel('time (sec)')
% ylabel('Velocity')
% set(gca,'FontSize',20)

% figure()
% x_time = 0.008*(1:1:length(velocity{1,1,1}));
% plot(x_time,velocity{1,1,1})
% grid on
% xlabel('time (sec)')
% ylabel('|Velocity|')
% set(gca,'FontSize',20)

% Step 3: Partial sum of velocity, create window dataset
vel_accum_gesture = cell(size(velocity));
vel_accum = {};
for i=1:size(input,3)
    for j=1:size(input,1)
        accum_tmp = [];
        for k=1:sliding:1+sliding*(fix((size(velocity{j,1,i},2)-window_size)/sliding + 1)-1)
            accum_tmp = [accum_tmp,sum(velocity{j,1,i}(k:k+window_size-1))];
        end
        vel_accum_gesture{j,1,i} = accum_tmp;
        vel_accum = [vel_accum;accum_tmp];
    end
end

% Step 4: Offset compensation & accumulated sum of partial sum
vel_compen_gesture = cell(size(vel_accum_gesture));
for i=1:size(vel_accum_gesture,3)
    for j=1:size(vel_accum_gesture,1)
        num_sample = fix(size(vel_accum_gesture{j,1,i},2)*0.1);
        offset = mean([vel_accum_gesture{j,1,i}(1:num_sample),vel_accum_gesture{j,1,i}(end-(num_sample-1):end)]);
        vel_compen_gesture{j,1,i} = vel_accum_gesture{j,1,i} - offset;    
    end
end
vel_compen = cell(size(vel_accum));
for i=1:size(vel_accum,1)
    num_sample = fix(size(vel_accum{i,1},2)*0.1);
    offset = mean([vel_accum{i,1}(1:num_sample),vel_accum{i,1}(end-(num_sample-1):end)]);
    vel_compen{i,1} = vel_accum{i,1} - offset;
end
% figure()
% x_time = 0.008*(1:1:length(vel_accum{1,1}));
% plot(x_time,vel_accum{1,1})
% grid on
% xlabel('time (sec)')
% ylabel('Partial sum of vel')
% set(gca,'FontSize',20)

pos_accum_gesture = cell(size(vel_compen_gesture));
for i=1:size(vel_compen_gesture,3)
    for j=1:size(vel_compen_gesture,1)
        pos_tmp = [];
        for k=1:size(vel_compen_gesture{j,1,i},2)
            pos_tmp = [pos_tmp,sum(vel_compen_gesture{j,1,i}(:,1:k))];
        end
        pos_accum_gesture{j,1,i} = pos_tmp;
    end
end
pos_accum = {};
for i=1:size(vel_compen,1)
    pos_tmp = [];
    for j=1:size(vel_compen{i,1},2)
        pos_tmp = [pos_tmp,sum(vel_compen{i,1}(:,1:j))];
    end
    pos_accum = [pos_accum;pos_tmp];
end
% figure()
% x_time = 0.008*(1:1:length(pos_accum{1,1}));
% plot(x_time,pos_accum{1,1})
% grid on
% xlabel('time (sec)')
% ylabel('Accumulated sum')
% set(gca,'FontSize',20)

% Step 5: Normalize both vel_compen & pos_accum
vel_compen_norm = cell(size(vel_compen));
pos_accum_norm = cell(size(pos_accum));
for i=1:size(vel_compen,1)
    vel_compen_norm{i,1} = (vel_compen{i,1} - min(vel_compen{i,1}))/(max(vel_compen{i,1}) - min(vel_compen{i,1}));
    pos_accum_norm{i,1} = (pos_accum{i,1} - min(pos_accum{i,1}))/(max(pos_accum{i,1}) - min(pos_accum{i,1}));
end
vel_compen_norm_gesture = cell(size(vel_compen_gesture));
pos_accum_norm_gesture = cell(size(pos_accum_gesture));
for i=1:size(vel_compen_gesture,3)
    for j=1:size(vel_compen_gesture,1)
        vel_compen_norm_gesture{j,1,i} = (vel_compen_gesture{j,1,i} - min(vel_compen_gesture{j,1,i}))/(max(vel_compen_gesture{j,1,i}) - min(vel_compen_gesture{j,1,i}));
        pos_accum_norm_gesture{j,1,i} = (pos_accum_gesture{j,1,i} - min(pos_accum_gesture{j,1,i}))/(max(pos_accum_gesture{j,1,i}) - min(vel_compen_gesture{j,1,i}));
    end
end
% figure()
% x_time = 0.008*(1:1:length(vel_compen_norm_gesture{1,1,1}));
% plot(x_time,vel_compen_norm_gesture{1,1,1})
% hold on
% plot(x_time,pos_accum_norm_gesture{1,1,1})
% grid on
% xlabel('time (sec)')
% ylabel('Normalization')
% set(gca,'FontSize',20)

% Step 6: GPS
GPS = cell(size(vel_compen_norm));
for i=1:size(vel_compen_norm,1)
    GPS{i,1} = (1-vel_compen_norm{i,1}).*pos_accum_norm{i,1} + vel_compen_norm{i,1}*1/2;
end
GPS_gesture = cell(size(vel_compen_norm_gesture));
for i=1:size(vel_compen_norm_gesture,3)
    for j=1:size(vel_compen_norm_gesture,1)
        GPS_gesture{j,1,i} = (1-vel_compen_norm_gesture{j,1,i}).*pos_accum_norm_gesture{j,1,i} + vel_compen_norm_gesture{j,1,i}*1/2;
    end
end
% figure()
% x_time = 0.008*(1:1:length(GPS_gesture{1,1,1}));
% plot(x_time,GPS_gesture{1,1,1})
% grid on
% xlabel('time (sec)')
% ylabel('GPS')
% set(gca,'FontSize',20)

% Construct output
output_window_data = {};
for i=1:size(input,3)
    for j=1:size(input,1)
        for k=1:sliding:1+sliding*(fix((size(input{j,1,i},2)-window_size)/sliding + 1)-1)
            output_window_data = [output_window_data;input{j,1,i}(:,k:k+window_size-1)'];
        end
    end
end

output_GPS = {};
for i=1:size(GPS,1)
    for j=1:size(GPS{i,1},2)
        output_GPS = [output_GPS;GPS{i,1}(:,j)];
    end
end

output = [output_window_data, output_GPS];

end