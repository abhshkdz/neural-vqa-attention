require 'hdf5'
cjson = require 'cjson'
utils = require 'misc/utils'

local dataloader = {}

function dataloader:initialize(opt, split)
    print('Reading ' .. opt.params_json)
    local file = io.open(opt.params_json, 'r')
    local text = file:read()
    file:close()
    local params = cjson.decode(text)
    for k,v in pairs(params) do self[k] = v end
    self['vocab_size'] = 0 for i,w in pairs(self['ix_to_word']) do self['vocab_size'] = self['vocab_size'] + 1 end

    print('Reading ' .. opt['img_' .. split .. '_h5'])
    local img_data = hdf5.open(opt['img_' .. split .. '_h5'], 'r')
    self['fv_im'] = img_data:read('/images_' .. split):all()
    img_data:close()

    print ('Reading ' .. opt.qa_h5)
    local qa_data = hdf5.open(opt.qa_h5, 'r')
    -- image
    self['im_list'] = qa_data:read('/img_pos_' .. split):all()
    -- question
    self['ques'] = qa_data:read('/ques_' .. split):all()
    self['ques_len'] = qa_data:read('ques_length_' .. split):all()
    self['ques'] = utils.right_align(self['ques'], self['ques_len'])
    -- answer
    if split == 'train' then
        self['ans'] = qa_data:read('/answers'):all()
    end
    qa_data:close()
end

function dataloader:next_batch(opt)
    local iminds = torch.LongTensor(opt.batch_size):fill(0)
    local qinds = torch.LongTensor(opt.batch_size):fill(0)

    for i = 1, opt.batch_size do
        qinds[i] = torch.random(dataloader['ques']:size(1))
        iminds[i] = self['im_list'][qinds[i]]
    end

    im = self['fv_im']:index(1, iminds)
    ques = self['ques']:index(1, qinds)
    labels = self['ans']:index(1, qinds)

    if opt.gpu >= 0 then
        im = im:cuda()
        ques = ques:cuda()
        labels = labels:cuda()
    end

    return {im, ques, labels}
end

return dataloader
