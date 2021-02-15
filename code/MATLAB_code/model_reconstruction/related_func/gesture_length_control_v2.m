function output = gesture_length_control_v2(input,window_size)
% input: 샘플 수 * 1 * 제스처 클래스
% output: 샘플 수 * 1 * 제스처 클래스
%% Velocity 계산 및 제스처 진행 척도 계산
% Velocity 계산
% low-pass filter
Fs = 100; % sampling freq
n = 2; % 차수
Wn = 4; % cut off freq
Fn = Fs/2; % Nyquist freq
ftype = 'low';
[b,a] = butter(n,Wn/Fn, ftype);
velocity_training = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        pc = pca(input{i,1,j}');
        pc1 = pc(:,1);
        pc1_mag = norm(pc1);
        principal_gesture = [];
        for k=1:size(input{i,1,j},2)
            principal_gesture = [principal_gesture,dot(input{i,1,j}(:,k),pc1)/pc1_mag];
        end
        for k=1:size(input{i,1,j},2)-1
            velocity_training{i,1,j}(:,k) = principal_gesture(:,k+1) - principal_gesture(:,k);
        end
        velocity_training{i,1,j} = filter(b,a,velocity_training{i,1,j});
        tmp_offset = (mean(velocity_training{i,1,j}(:,1:20),2) + mean(velocity_training{i,1,j}(:,end-19:end),2))/2;
        velocity_training{i,1,j} = velocity_training{i,1,j}-tmp_offset*ones(1,size(velocity_training{i,1,j},2));
        velocity_training{i,1,j} = abs(velocity_training{i,1,j});
        % velocity 제곱함. (제스처 단계 구분 명확히 하기 위해)
        velocity_training{i,1,j} = velocity_training{i,1,j}.*velocity_training{i,1,j};
        velocity_training{i,1,j}(1,1:10) = 0;
    end
end
%% Velocity 플랏
g_num = 14;
figure()
for i=1+5*(g_num-1):5*g_num
    plot(velocity_training{i,1})
    hold on
end
grid on
%% 제스처 단계 별 길이 통제
phase_length_org = [];
feature_point = [];
for i=1:size(input,1)
    for j=1:size(input,3)
        feature_point(i,1,j) = 1;
        
        [max_vel,max_vel_loc] = max(velocity_training{i,1,j});
        [~,locs_max] = findpeaks(velocity_training{i,1,j});
        
        locs_max = locs_max(locs_max > max_vel_loc*0.5); % 좌측에 작은 peak 있을 경우 고려
        locs_max = locs_max(locs_max < length(velocity_training{i,1,j}) - max_vel_loc*0.5); % 우측에 작은 peak 있을 경우 고려
        
        candidate = locs_max(velocity_training{i,1,j}(locs_max) > max_vel*0.2); % percentage threshold 0.2
        first_peak = candidate(1);
        last_peak = candidate(end);
        
        locs_max_left_portion = locs_max(locs_max < first_peak);
        while isempty(find(velocity_training{i,1,j}(locs_max_left_portion) > 1,1)) == 0
            first_peak_cand = locs_max_left_portion(velocity_training{i,1,j}(locs_max_left_portion) > 1); % velocity threshold 1
            first_peak = first_peak_cand(end);
            locs_max_left_portion = locs_max(locs_max < first_peak);
        end
        for k=first_peak:-1:1
            if velocity_training{i,1,j}(k) < 0.5 % velocity threshold 0.5
                feature_point(i,2,j) = k;
                break
            end
        end
        
        locs_max_right_portion = locs_max(locs_max > last_peak);
        while isempty(find(velocity_training{i,1,j}(locs_max_right_portion) > 1,1)) == 0
            last_peak_cand = locs_max_right_portion(velocity_training{i,1,j}(locs_max_right_portion) > 1); % velocity threshold 1
            last_peak = last_peak_cand(1);
            locs_max_right_portion = locs_max(locs_max > last_peak);
        end
        for k=last_peak:length(velocity_training{i,1,j})
            if velocity_training{i,1,j}(k) < 0.5 % velocity threshold 0.5
                feature_point(i,3,j) = k;
                break
            end
        end
        
        feature_point(i,4,j) = size(input{i,1,j},2);

        phase_length_org(i,1,j) = feature_point(i,2,j) - feature_point(i,1,j) + 1;
        phase_length_org(i,2,j) = feature_point(i,3,j) - feature_point(i,2,j) - 1;
        phase_length_org(i,3,j) = feature_point(i,4,j) - feature_point(i,3,j) + 1;
    end
end

phase_length_cont = [];
for i=1:size(phase_length_org,1)
    for j=1:size(phase_length_org,3)
        % 초기 길이를 중기와 동일하게 맞추기
        if phase_length_org(i,1,j) < phase_length_org(i,2,j) % 초기 길이가 중기 길이보다 짧을 경우
            phase_length_cont(i,1,j) = phase_length_org(i,1,j);
        else
            % 중기 길이가 초기 길이보다 짧을 경우
            phase_length_cont(i,1,j) = phase_length_org(i,2,j);
        end
        
        % 중기 길이는 그대로
        phase_length_cont(i,2,j) = phase_length_org(i,2,j);
        
        % 후기 길이를 중기와 동일하게 맞추기
        if phase_length_org(i,3,j) < phase_length_org(i,2,j) % 후기 길이가 중기 길이보다 짧을 경우
            phase_length_cont(i,3,j) = phase_length_org(i,3,j);
        else
            % 중기 길이가 후기 길이보다 짧을 경우
            phase_length_cont(i,3,j) = phase_length_org(i,2,j);
        end
        
        % 초기 길이 최종 조정
        if phase_length_cont(i,1,j) < window_size % 조정된 길이가 window size보다 짧을 경우
            phase_length_cont(i,1,j) = 2*window_size;
        else
            phase_length_cont(i,1,j) = phase_length_cont(i,1,j);
        end
        
        % 후기 길이 최종 조정
        if phase_length_cont(i,3,j) < window_size % 조정된 길이가 window size보다 짧을 경우
            phase_length_cont(i,3,j) = 2*window_size;
        else
            phase_length_cont(i,3,j) = phase_length_cont(i,3,j);
        end        
    end
end

% test = {};
% prob = [];
% for i=1:size(input,1)
%     for j=1:size(input,3)
%         test{i,j} = [feature_point(i,2,j)-phase_length_cont(i,1,j)+1, feature_point(i,3,j)+phase_length_cont(i,3,j)-1];
%         if (test{i,j}(1) < 0) || (test{i,j}(2) > size(input{i,1,j},2))
%             prob = [prob,i];
%         end
%     end
% end
% 
% test2 = [];
% for i=1:size(input,1)
%     for j=1:size(input,3)
%         test2(i,j) = test{i,j}(2) - test{i,j}(1);
%     end
% end
% test3 = find(test2<0);
% 
% for i=prob
%     figure(i)
%     plot(velocity_training{i,1})
%     grid on
% end

output = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        output{i,1,j} = input{i,1,j}(:,feature_point(i,2,j)-phase_length_cont(i,1,j)+1:feature_point(i,3,j)+phase_length_cont(i,3,j)-1);
%         output{i,1,j} = input{i,1,j};
    end
end
end