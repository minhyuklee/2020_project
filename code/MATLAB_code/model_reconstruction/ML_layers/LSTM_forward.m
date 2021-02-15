function [hidden_set,interm_param] = LSTM_forward(input,weight,bias)
% Dimension
% input: (timestep,features)
% weight{:,1}: (features, hidden_unit)
% weight{:,2}: (hidden_unit, hidden_unit)
% bias{:,:}: (hidden_unit, 1)
% hidden_set: (timestep,hidden_unit)

forgetG = zeros(1,size(weight{1,1},2));
inputG = zeros(1,size(weight{2,1},2));
gG = zeros(1,size(weight{3,1},2));
outputG = zeros(1,size(weight{4,1},2));
state = zeros(1,size(weight{1,1},2));
tc = zeros(1,size(weight{1,1},2));
hidden = zeros(1,size(weight{1,2},2));

forgetGmat = forgetG;
inputGmat = inputG;
gGmat = gG;
outputGmat = outputG;
stateGmat = state;
tcmat = tc;
hidden_set = hidden;

for t = 1:size(input,1)
    forgetG = sigmf(hidden*weight{1,2} + input(t,:)*weight{1,1} + bias{1,1}',[1 0]);
    inputG = sigmf(hidden*weight{2,2} + input(t,:)*weight{2,1} + bias{2,1}',[1 0]);
    gG = tanh(hidden*weight{3,2} + input(t,:)*weight{3,1} + bias{3,1}');
    outputG = sigmf(hidden*weight{4,2} + input(t,:)*weight{4,1} + bias{4,1}',[1 0]);
    state = forgetG.*state + inputG.*gG;
    tc = tanh(state);
    hidden = tc.*outputG;

    forgetGmat = [forgetGmat;forgetG];
    inputGmat = [inputGmat;inputG];
    gGmat = [gGmat;gG];
    outputGmat = [outputGmat;outputG];
    stateGmat = [stateGmat;state];
    tcmat = [tcmat;tc];
    hidden_set = [hidden_set;hidden];
end
interm_param = cell(1,6);
interm_param{1,1} = forgetGmat;
interm_param{1,2} = inputGmat;
interm_param{1,3} = gGmat;
interm_param{1,4} = outputGmat;
interm_param{1,5} = stateGmat;
interm_param{1,6} = tcmat; 
end