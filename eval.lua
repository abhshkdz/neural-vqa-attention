require 'torch'
require 'nn'
require 'rnn'
san = require 'misc/san'
cjson = require 'cjson'

opt = {
    -- data
    img_train_h5 = 'data/img_train.h5',
    img_test_h5 = 'data/img_test.h5',
    qa_h5 = 'data/qa.h5',
    params_json = 'data/params.json',

    -- model
    model_path = 'checkpoints/model.t7',

    -- misc
    batch_size = 500,
    gpu = 0,
    seed = 123,
    result_path = 'results/'
}

-- one-line argument parser. parses enviroment variables to override the defaults
-- from https://github.com/soumith/dcgan.torch/blob/master/main.lua#L25
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
paths.mkdir(opt.result_path)
if opt.result_path == 'results/' then
    local cur_time = os.date('*t', os.time())
    opt.result_path = string.format('results/%d-%d-%d-%d:%d:%d/', cur_time.month, cur_time.day, cur_time.year, cur_time.hour, cur_time.min, cur_time.sec)
end
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- dataloader
dataloader = dofile('dataloader.lua')
dataloader:initialize(opt, 'test')
collectgarbage()

print('Vocab size ' .. dataloader.vocab_size)

if opt.gpu >= 0 then
    require 'cutorch'
    require 'cunn'
end

-- loading from checkpoint
saved = torch.load(opt.model_path)

-- model
lm = nn.Sequential()
        :add(nn.LookupTableMaskZero(dataloader.vocab_size, saved.opt.embedding_size))
        :add(nn.Dropout(0.5))
        :add(nn.SplitTable(1, 2))
        :add(nn.Sequencer(nn.FastLSTM(saved.opt.embedding_size, saved.opt.rnn_size):maskZero(1)))
        :add(nn.Sequencer(nn.FastLSTM(saved.opt.rnn_size, saved.opt.rnn_size):maskZero(1)))
        :add(nn.SelectTable(-1))

model = nn.Sequential()
            :add(nn.ParallelTable()
                :add(nn.Identity())
                :add(lm))
            :add(san.n_attention_layer(saved.opt))

if opt.gpu >= 0 then
    print('Shipping to cuda')
    model = model:cuda()
end

model_params, model_grad_params = model:getParameters()

-- copying params from checkpoint
model_params:copy(saved['model_params'])
model:evaluate()

-- softmax layers in san-k for k = 1, 2
attention_layers = {{19}, {22, 35}}

-- forward pass; returns scores and attention maps
function forward(batch)
    local scores = model:forward({batch[1], batch[2]})
    local mapify = nn.View(-1, 14, 14)
    local att_maps = {}

    for i = 1, saved.opt.num_attention_layers do
        table.insert(att_maps, mapify:forward(model.modules[2].modules[attention_layers[saved.opt.num_attention_layers][i]].output:float()))
    end

    return {scores:float(), att_maps}
end

-- eval loop
local num_batches = math.ceil(dataloader['ques']:size(1) / opt.batch_size)
print('No. batches ' .. num_batches)

local ques_id = torch.LongTensor(dataloader['ques']:size(1)):fill(0)
local scores = torch.zeros(dataloader['ques']:size(1), saved.opt.num_output)
local maps = torch.zeros(saved.opt.num_attention_layers, dataloader['ques']:size(1), 14, 14)

for i = 1, num_batches do
    xlua.progress(i, num_batches)
    local start_id = dataloader.test_id
    local batch = dataloader:next_batch_eval(opt)
    local end_id = dataloader.test_id - 1

    output = forward(batch)

    ques_id[{{start_id, end_id}}] = batch[3]
    scores[{{start_id, end_id}, {}}] = output[1]
    for j = 1, saved.opt.num_attention_layers do
        maps[{{j}, {start_id, end_id}, {}}] = output[2][j]
    end
end

_, preds = scores:max(2)

-- from https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/eval.lua#L230 onwards
function write_json(file, data)
    print('Writing to ' .. file)
    local f = io.open(file, 'w')
    f:write(cjson.encode(data))
    f:close()
end

paths.mkdir(opt.result_path)
local oe_res = {}
for i = 1, dataloader['ques']:size(1) do
    table.insert(oe_res, {question_id = ques_id[i], answer = dataloader['ix_to_ans'][tostring(preds[{i, 1}])]})
end
write_json(opt.result_path .. 'OpenEnded_mscoco_lstm_results.json', oe_res)

-- mc_res = {}
-- for i = 1, dataloader['ques']:size(1) do
    -- local mc_prob = {}
    -- local tmp_id = {}
    -- for j = 1, dataloader['ans_mc'][i]:size(1) do
        -- if dataloader['ans_mc'][i][j] ~= 0 then
            -- table.insert(mc_prob, scores[{i, dataloader['ans_mc'][i][j]}])
            -- table.insert(tmp_id, dataloader['ans_mc'][i][j])
        -- end
    -- end
    -- local _, tmp = torch.max(torch.Tensor(mc_prob), 1)
    -- table.insert(mc_res, {question_id = ques_id[i], answer = dataloader['ix_to_ans'][tostring(tmp_id[tmp[1]])]})
-- end
-- write_json(opt.result_path .. 'MultipleChoice_mscoco_lstm_results.json', mc_res)

print('Writing to ' .. opt.result_path .. 'attention_maps.h5')
local att_h5 = hdf5.open(opt.result_path .. 'attention_maps.h5', 'w')
att_h5:write('/maps', maps)
att_h5:close()

