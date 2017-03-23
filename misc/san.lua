require 'nn'
require 'nngraph'

local san = {}

function san.n_attention_layer(opt)
    local inputs, outputs = {}, {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local img_feat = inputs[1]
    local ques_feat = inputs[2]

    local u = ques_feat
    local img_tr = nn.Dropout(0.5)(nn.Tanh()(nn.View(-1, 196, opt.im_tr_size)(nn.Linear(512, opt.im_tr_size)(nn.View(512):setNumInputDims(2)(img_feat)))))

    for i = 1, opt.num_attention_layers do

        -- linear layer: 14x14x1024 -> 14x14x512
        local img_common = nn.View(-1, 196, opt.common_embedding_size)(nn.Linear(opt.im_tr_size, opt.common_embedding_size)(nn.View(-1, opt.im_tr_size)(img_tr)))

        -- replicate lstm state 196 times
        local ques_common = nn.Linear(opt.rnn_size, opt.common_embedding_size)(u)
        local ques_repl = nn.Replicate(196, 2)(ques_common)

        -- add image and question features (both 196x512)
        local img_ques_common = nn.Dropout(0.5)(nn.Tanh()(nn.CAddTable()({img_common, ques_repl})))
        local h = nn.Linear(opt.common_embedding_size, 1)(nn.View(-1, opt.common_embedding_size)(img_ques_common))
        local p = nn.SoftMax()(nn.View(-1, 196)(h))

        -- weighted sum of image features
        local p_att = nn.View(1, -1):setNumInputDims(1)(p)
        local img_tr_att = nn.MM(false, false)({p_att, img_tr})
        local img_tr_att_feat = nn.View(-1, opt.im_tr_size)(img_tr_att)
        -- add image feature vector and question vector
        u = nn.CAddTable()({img_tr_att_feat, u})

    end

    -- MLP to answers
    local o = nn.LogSoftMax()(nn.Linear(opt.rnn_size, opt.num_output)(nn.Dropout(0.5)(u)))

    table.insert(outputs, o)

    return nn.gModule(inputs, outputs)
end

return san
