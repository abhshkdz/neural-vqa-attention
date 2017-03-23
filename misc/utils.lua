cjson = require 'cjson'

local utils = {}

function utils.right_align(sequences, lengths)
    local aligned = sequences:clone():fill(0)

    local n = sequences:size(1) -- number of rows
    local m = sequences:size(2) -- maximum length

    for i = 1, n do
        if lengths[i] > 0 then
            aligned[i][{{m - lengths[i] + 1, m}}] = sequences[i][{{1, lengths[i]}}]
        end
    end

    return aligned
end

return utils
