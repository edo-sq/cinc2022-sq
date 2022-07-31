function model = team_training_code(input_directory,output_directory) % train_PCG_classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Train PCG classifiers and obtain the models
% Inputs:
% 1. input_directory
% 2. output_directory
%
% Outputs:
% 1. model: trained model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find text files
patient_files=dir(fullfile(input_directory,'*.txt'));
patient_files={patient_files.name};
patient_files=sort(patient_files); % To help debugging
num_patient_files=length(patient_files);

fprintf('Loading data for %d patients...\n', num_patient_files)

% Extract classes from data
classes_murmur={};
classes_outcome={};
for j=1:num_patient_files

    current_class_murmur=get_class_murmur(fullfile(input_directory,patient_files{j}));
    classes_murmur=unique([classes_murmur current_class_murmur]);

    current_class_outcome=get_class_outcome(fullfile(input_directory,patient_files{j}));
    classes_outcome=unique([classes_outcome current_class_outcome]);

end

classes_murmur=sort(classes_murmur);
num_classes_murmur=length(classes_murmur);

classes_outcome=sort(classes_outcome);
num_classes_outcome=length(classes_outcome);

% Extracting features and labels
disp('Extracting features and labels...')

features=[];
labels_murmur=categorical;
labels_outcome=categorical;

data_tbl = table();

for j=1:num_patient_files

    fprintf('%d/%d \n',j,num_patient_files)
    tmp = table();

    current_header=get_header(fullfile(input_directory,patient_files{j}));
    h1_split = split(string(current_header{1}));
    tmp.PatientID = h1_split(1);
    
    tmp.fs = h1_split(3);
    current_recordings=load_recordings(input_directory,current_header);

    rec_locs = "";
    
    recs = [];
    for i = 1:double(h1_split(2))
        rec = struct();
        loc = split(string(current_header(1+i)));
        loc = loc(1);
        rec.loc = loc;
        if rec_locs == ""
            rec_locs = loc;
        else
            rec_locs = rec_locs + "+" + loc;
        end
        rec.data = sq_preprocess(current_recordings{i}, 8300, 5000);
        recs = [recs; rec];
    end
    tmp.recs = {recs};
    tmp.locs = rec_locs;

    tmp.murmur = string(get_class_murmur(fullfile(input_directory,patient_files{j})));
    tmp.outcome = string(get_class_outcome(fullfile(input_directory,patient_files{j})));
    data_tbl = [data_tbl; tmp];
end

%% train RF

disp('Training the model...')

locs = ["AV", "MV", "TV", "PV"];

murmur_classes = categories(categorical(data_tbl(data_tbl.murmur == "Present" | data_tbl.murmur == "Absent", : ).murmur));
outcome_classes = categories(categorical(data_tbl(data_tbl.outcome == "Abnormal" | data_tbl.outcome == "Normal", : ).outcome));

murmur_models = struct();
outcome_models = struct();

for i = 1:length(locs)
    loc = locs(i);

    dt_present = data_tbl(data_tbl.murmur == "Present", :);
    dt_absent = data_tbl(data_tbl.murmur == "Absent", :);

    [train_mp, val_mp] = tv_split(dt_present, 0.1);
    [train_ma, val_ma] = tv_split(dt_absent, 0.1);

    train_mp_x = extract_beats(train_mp, loc);
    train_ma_x = extract_beats(train_ma, loc);

    val_mp_x = extract_beats(val_mp, loc);
    val_ma_x = extract_beats(val_ma, loc);

    train_x = cat(4, train_mp_x, train_ma_x);
    train_y = categorical([repmat("Present", size(train_mp_x, 4), 1); repmat("Absent", size(train_ma_x, 4), 1)]);
    train_ds = arrayDatastore([arrayfun(@(i) train_x(:,:,:,i), 1:size(train_x, 4), 'uni', 0)', mat2cell(train_y, ones(length(train_y), 1), 1)], 'OutputType', 'same');

    val_x = cat(4, val_mp_x, val_ma_x);
    val_y = categorical([repmat("Present", size(val_mp_x, 4), 1); repmat("Absent", size(val_ma_x, 4), 1)]);
    val_dla = dlarray(val_x, 'SSCB');
    
    murmur_models.(loc) = sq_train(train_ds, val_dla, val_y, sq_cnn(train_x), murmur_classes);
end

for i = 1:length(locs)
    loc = locs(i);

    dt_present = data_tbl(data_tbl.outcome == "Abnormal", :);
    dt_absent = data_tbl(data_tbl.outcome == "Normal", :);

    [train_mp, val_mp] = tv_split(dt_present, 0.1);
    [train_ma, val_ma] = tv_split(dt_absent, 0.1);

    train_mp_x = extract_beats(train_mp, loc);
    train_ma_x = extract_beats(train_ma, loc);

    val_mp_x = extract_beats(val_mp, loc);
    val_ma_x = extract_beats(val_ma, loc);

    train_x = cat(4, train_mp_x, train_ma_x);
    train_y = categorical([repmat("Abnormal", size(train_mp_x, 4), 1); repmat("Normal", size(train_ma_x, 4), 1)]);
    train_ds = arrayDatastore([arrayfun(@(i) train_x(:,:,:,i), 1:size(train_x, 4), 'uni', 0)', mat2cell(train_y, ones(length(train_y), 1), 1)], 'OutputType', 'same');

    val_x = cat(4, val_mp_x, val_ma_x);
    val_y = categorical([repmat("Abnormal", size(val_mp_x, 4), 1); repmat("Normal", size(val_ma_x, 4), 1)]);
    val_dla = dlarray(val_x, 'SSCB');
    
    outcome_models.(loc) = sq_train(train_ds, val_dla, val_y, sq_cnn(train_x), outcome_classes);
end



save_model(murmur_models,murmur_classes,outcome_models,outcome_classes,output_directory);

disp('Done.')

end

function [res] = extract_beats(data, loc)
    loc_data = data(contains(string(data.locs), loc), :);
    res_c = [];
    for i = 1:height(loc_data)
        d = loc_data(i, :);
        recs = d.recs{1};
        res_c = [res_c; cat(1, recs(contains([recs.loc], loc)).data)];
    end
    res = cat(4, res_c{:});
end

function save_model(model_murmur,classes_murmur,model_outcome,classes_outcome,output_directory) %save_PCG_model
% Save results.
filename = fullfile(output_directory,'model.mat');
save(filename,'model_murmur','classes_murmur','model_outcome','classes_outcome','-v7.3');

disp('Done.')
end

function class=get_class_murmur(input_header)

current_header=get_header(input_header);

class=current_header(startsWith(current_header,'#Murmur'));
class=strsplit(class{1},':');
class=strtrim(class{2});

end

function class=get_class_outcome(input_header)

current_header=get_header(input_header);

class=current_header(startsWith(current_header,'#Outcome'));
class=strsplit(class{1},':');
class=strtrim(class{2});

end

function current_header=get_header(input_header)

current_header=fileread(input_header);
current_header=strsplit(current_header,'\n');

end

function current_recordings=load_recordings(input_directory,current_header)

recording_files=get_recording_files(current_header);

current_recordings={};

for j=1:length(recording_files)

    current_recordings{j}=audioread(fullfile(input_directory,strtrim(recording_files{j})));

end

end

function recording_files=get_recording_files(current_header)

recording_files={};

num_locations=strsplit(current_header{1},' ');
num_locations=str2double(num_locations{2});

for j=2:num_locations+1

    current_line=strsplit(current_header{j},' ');
    recording_files{j-1}=current_line{3};

end

end

function [train, val] = tv_split(data, vr)
    H = height(data);
    vc = round(vr * H);
    

    val_idx = randperm(H, vc);

    val = data(val_idx, :);

    train = data;
    train(val_idx,:) = [];
end

