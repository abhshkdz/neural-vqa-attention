require 'nn'
require 'xlua'
require 'math'
require 'hdf5'
require 'image'
require 'loadcaffe'
cjson = require('cjson')

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-input_json','data/params.json','path to the json file containing vocab and answers')
cmd:option('-image_root','/path/to/coco/images/','path to the image root')
cmd:option('-cnn_proto', 'models/vgg19/VGG_ILSVRC_19_layers_deploy.prototxt', 'path to the cnn prototxt')
cmd:option('-cnn_model', 'models/vgg19/VGG_ILSVRC_19_layers.caffemodel', 'path to the cnn model')
cmd:option('-batch_size', 20, 'batch_size')

cmd:option('-out_name_train', 'data/img_train.h5', 'output name')
cmd:option('-out_name_test', 'data/img_test.h5', 'output name')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-backend', 'nn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

net = loadcaffe.load(opt.cnn_proto, opt.cnn_model, opt.backend);
for i = 1, 9 do
    net:remove()
end
print(net)

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid+1)
    net = net:cuda()
end
net:evaluate()

function loadim(imname)
    im = image.load(imname)
    im = image.scale(im,448,448)
    if im:size(1) == 1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1) == 4 then
        im=im[{{1,3},{},{}}]
    end
    im = im * 255;
    im2 = im:clone()
    im2[{{3},{},{}}] = im[{{1},{},{}}]-123.68
    im2[{{2},{},{}}] = im[{{2},{},{}}]-116.779
    im2[{{1},{},{}}] = im[{{3},{},{}}]-103.939
    return im2
end

local image_root = opt.image_root

local file = io.open(opt.input_json, 'r')
local text = file:read()
file:close()
json_file = cjson.decode(text)

local train_list={}
for i,imname in pairs(json_file['unique_img_train']) do
    table.insert(train_list, image_root .. imname)
end

local test_list={}
for i,imname in pairs(json_file['unique_img_test']) do
    table.insert(test_list, image_root .. imname)
end

local batch_size = opt.batch_size
local sz=#train_list
local feat_train=torch.FloatTensor(sz, 14, 14, 512)
print(string.format('processing %d images...',sz))
for i = 1, sz, batch_size do
    xlua.progress(i, sz)
    r = math.min(sz, i+batch_size-1)
    ims = torch.DoubleTensor(r-i+1, 3, 448,448)
    for j = 1, r-i+1 do
        ims[j] = loadim(train_list[i+j-1])
    end
    if opt.gpuid >= 0 then
        ims = ims:cuda()
    end
    net:forward(ims)
    feat_train[{{i,r}, {}}] = net.output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end

local train_h5_file = hdf5.open(opt.out_name_train, 'w')
train_h5_file:write('/images_train', feat_train)
train_h5_file:close()

print('DataLoader loading h5 file: ', 'data_train')
local sz = #test_list
local feat_test = torch.FloatTensor(sz, 14, 14, 512)
print(string.format('processing %d images...',sz))
for i = 1, sz, batch_size do
    xlua.progress(i, sz)
    r = math.min(sz, i + batch_size-1)
    ims = torch.DoubleTensor(r-i+1, 3, 448, 448)
    for j = 1, r-i+1 do
        ims[j] = loadim(test_list[i+j-1])
    end
    if opt.gpuid >= 0 then
        ims = ims:cuda()
    end
    net:forward(ims)
    feat_test[{{i,r}, {}}] = net.output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end

local test_h5_file = hdf5.open(opt.out_name_test, 'w')
test_h5_file:write('/images_test', feat_test)
test_h5_file:close()

