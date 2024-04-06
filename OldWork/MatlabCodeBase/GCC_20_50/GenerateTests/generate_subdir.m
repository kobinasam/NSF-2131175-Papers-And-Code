function subdir = generate_subdir(condition_labels, conditions)

    subdir = '';
    for i=1: size(condition_labels, 2)
        if conditions(i) == 1
            subdir = strcat(subdir, 'yes');
        else
            subdir = strcat(subdir, 'no');
        end
        subdir = strcat(subdir, '_', condition_labels(i), '_');
    end


    subdir = convertStringsToChars(subdir);
    if ~isempty(subdir)
        % get rid of last underscore
        subdir = subdir(1:end-1);
    end
end