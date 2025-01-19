#Chat gpt replacement for StatsBase

function countmap(collection)
    counts = Dict{eltype(collection), Int}()
    for item in collection
        counts[item] = get(counts, item, 0) + 1
    end
    return counts
end


function mode(collection)
    # Erstelle ein Countmap
    counts = countmap(collection)
    
    # Finde das häufigste Element
    max_count = -1
    mode_element = nothing
    for (key, value) in counts
        if value > max_count
            max_count = value
            mode_element = key
        end
    end
    
    return mode_element
end

function mean(collection)
    return sum(collection) / length(collection)
end