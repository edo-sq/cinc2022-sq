%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Run trained PCG classifier and obtain classifier outputs
% Inputs:
% 1. header
% 2. recordings
% 3. trained model
%
% Outputs:
% 1. score
% 2. label
% 3. classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [score, label, classes] = team_testing_code(header, recordings, loaded_model)

classes_murmur=loaded_model.classes_murmur;
classes_outcome=loaded_model.classes_outcome;

% Find number of recordings.
h1_split = split(string(header{1}));
rec_num = double(h1_split(2));
m_res = [0.5;0.5];
o_res = [0.5;0.5];
for i = 1:rec_num
    loc = split(string(header(1+i)));
    loc = loc(1);
    d = sq_preprocess(recordings{i}, 8300, 5000);

    tmp_m_res = sum(predict(loaded_model.model_murmur.(loc), dlarray(cat(4, d{:}), 'SSCB')),2);
    tmp_m_res = tmp_m_res/sum(tmp_m_res);
    m_res = m_res + tmp_m_res;

    tmp_o_res = sum(predict(loaded_model.model_outcome.(loc), dlarray(cat(4, d{:}), 'SSCB')),2);
    tmp_o_res = tmp_o_res/sum(tmp_o_res);
    o_res = o_res + tmp_o_res;
end
o_res = o_res./ sum(o_res);
m_res = m_res./sum(m_res);

% fm_score = [0.0 0.0 0.0];
mp_idx = string(classes_murmur) == "Present";
if mp_idx == 0
    ma_idx = 1;
else
    ma_idx = 0;
end
fm_classes = ["Present", "Unknown", "Absent"];
fo_classes = string(classes_outcome');
if m_res(mp_idx) >= 0.25
    fm_score = [m_res(mp_idx) 0.0 m_res(ma_idx)];
    fm_labels = [1 0 0];
elseif m_res(mp_idx) < 0.25 && m_res(mp_idx) >= 0.15
    fm_score = [m_res(mp_idx), (m_res(mp_idx) + m_res(ma_idx))/2.0, m_res(ma_idx)];
    fm_score = fm_score./sum(fm_score);
    fm_labels = [0 1 0];
else
    fm_score = [m_res(mp_idx), 0.0, m_res(ma_idx)];
    fm_labels = [0 0 1];
end

classes=[fm_classes fo_classes];
score=[fm_score o_res];

label=[fm_labels onehotdecode(o_res, classes, 1)'];

end
