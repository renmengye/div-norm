clear;clc;

UP_FACTOR = 4;

TRAIN_DATASET = 'BSD200'
TRAIN_IMG_FMT = 'jpg'
TEST_DATASET = 'Set5'
TEST_IMG_FMT = 'bmp'
TRAIN_DATA_PATH = sprintf('sr_data/%s', TRAIN_DATASET);
TEST_DATA_PATH = sprintf('sr_data/%s', TEST_DATASET);
SAVE_PATH = sprintf('sr_data/data_X%d.h5', UP_FACTOR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Training Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_img_names = dir(fullfile(TRAIN_DATA_PATH, sprintf('*.%s', TRAIN_IMG_FMT)));

for ii = 1 : length(train_img_names)
    img = imread(fullfile(TRAIN_DATA_PATH, train_img_names(ii).name));
    
    if size(img, 3) > 1
        img = rgb2ycbcr(img);
        img = img(:, :, 1);
    end
    
    img = im2double(img);
    img = modcrop(img, UP_FACTOR);
    
    lr_img = imresize(imresize(img, 1/UP_FACTOR), UP_FACTOR);
    hr_img = img;
    
    dataset_name = sprintf('/lr_img/img_%06d', ii);
    h5create(SAVE_PATH, dataset_name, size(lr_img), 'Datatype', 'single');
    h5write(SAVE_PATH, dataset_name, single(lr_img));

    dataset_name = sprintf('/hr_img/img_%06d', ii);
    h5create(SAVE_PATH, dataset_name, size(hr_img), 'Datatype', 'single');
    h5write(SAVE_PATH, dataset_name, single(hr_img));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate Testing Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

test_img_names = dir(fullfile(TEST_DATA_PATH, sprintf('*.%s', TEST_IMG_FMT)));

for ii = 1 : length(test_img_names)
    img = imread(fullfile(TEST_DATA_PATH, test_img_names(ii).name));
    
    if size(img, 3) > 1
        img = rgb2ycbcr(img);
    end
    
    img = modcrop(img, UP_FACTOR);        

    if size(img, 3) > 1        
        img_y = img(:, :, 1);
    else
        img_y = img;
    end    
    
    lr_img_test = zeros(size(img));
    
    if size(img, 3) > 1
        lr_img_test(:, :, 1) = imresize(imresize(double(img_y) / 255.0, 1/UP_FACTOR), UP_FACTOR);
        lr_img_test(:, :, 2) = imresize(imresize(double(img(:, :, 2)), 1/UP_FACTOR), UP_FACTOR);
        lr_img_test(:, :, 3) = imresize(imresize(double(img(:, :, 3)), 1/UP_FACTOR), UP_FACTOR);
    else
        lr_img_test = imresize(imresize(double(img_y) / 255.0, 1/UP_FACTOR), UP_FACTOR);
    end
    
    hr_img_test = img;
    
    dataset_name = sprintf('/lr_img_test/img_%06d', ii);
    h5create(SAVE_PATH, dataset_name, size(lr_img_test), 'Datatype', 'single');
    h5write(SAVE_PATH, dataset_name, single(lr_img_test));

    dataset_name = sprintf('/hr_img_test/img_%06d', ii);
    h5create(SAVE_PATH, dataset_name, size(hr_img_test), 'Datatype', 'uint8');
    h5write(SAVE_PATH, dataset_name, hr_img_test);    
end
