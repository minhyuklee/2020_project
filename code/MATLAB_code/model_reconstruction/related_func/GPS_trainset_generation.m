function output = GPS_trainset_generation(input)
% input: 샘플 수 * 1 * 제스처 클래스
% output: 샘플 수 * 2 * 제스처 클래스
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
        for k=2:size(input{i,1,j},2)
            velocity_training{i,1,j}(:,k-1) = norm(input{i,1,j}(:,k) - input{i,1,j}(:,k-1));
        end
        velocity_training{i,1,j} = filter(b,a,velocity_training{i,1,j});
        tmp_offset = (mean(velocity_training{i,1,j}(:,1:20),2) + mean(velocity_training{i,1,j}(:,end-19:end),2))/2;
        velocity_training{i,1,j} = velocity_training{i,1,j}-tmp_offset*ones(1,size(velocity_training{i,1,j},2));
        velocity_training{i,1,j} = abs(velocity_training{i,1,j});
        % velocity 제곱함. (제스처 단계 구분 명확히 하기 위해)
%         velocity_training{i,1,j} = velocity_training{i,1,j}.*velocity_training{i,1,j};
        velocity_training{i,1,j}(1,1:10) = 0;
    end
end

% 제스처 진행 척도 계산
GPS = {}; % input과 길이 동일함.
for i=1:size(velocity_training,1)
    for j=1:size(velocity_training,3)
        integrat_sum = 0;
        integrat = integrat_sum;
        for k=1:size(velocity_training{i,1,j},2)
            integrat_sum = integrat_sum + velocity_training{i,1,j}(:,k);
            integrat = [integrat,integrat_sum];
        end
        integrat = 1/integrat(end)*integrat;
        GPS{i,1,j} = integrat;
    end
end

output = {};
for i=1:size(input,1)
    for j=1:size(input,3)
        output{i,1,j} = input{i,1,j};
        output{i,2,j} = GPS{i,1,j};
    end
end
%% Velocity, GPS 플랏
g_num = 1;
figure()
for i=1:size(GPS,1)
    plot(GPS{i,1,g_num},'LineWidth',2)
    hold on
end
xlabel('time step')
ylabel('GPS')
set(gca,'FontSize',16)
grid on

figure()
for i=1:size(velocity_training,1)
    plot(velocity_training{i,1,g_num},'LineWidth',2)
    hold on
end
xlabel('time step')
ylabel('absolute velocity')
set(gca,'FontSize',16)
grid on
end