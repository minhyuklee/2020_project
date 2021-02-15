function output = order_sub2class(dataset,subject_SE)
% subject_SE: 인풋 데이터셋에서 어떤 피험자 데이터부터(Start) 어떤 피험자 데이터까지(End) 저장할건지 결정
NumOfClass_Gesture = size(dataset{1,2,1},1); % 클래스 수
label_mat = eye(NumOfClass_Gesture);
class_tmp = cell(1,1,NumOfClass_Gesture);
NumDataforeachGesture = zeros(1,NumOfClass_Gesture);
for i=subject_SE(1):subject_SE(2)
    for j=1:size(dataset(:,:,i),1)
        for k=1:NumOfClass_Gesture
            if sum(dataset{j,2,i} == label_mat(:,k)) == NumOfClass_Gesture
                NumDataforeachGesture(:,k) = NumDataforeachGesture(:,k) + 1;
                class_tmp{NumDataforeachGesture(:,k),1,k} = dataset{j,1,i};
            end
        end
    end
end
output = class_tmp;
end