clearinfo

# this script extracts the utterances of all audio + TextGrid files in a directory

# modify input and output directory
input_dir$ = "/home/philipp/Workspace/test/"

output_dir$ = "/home/philipp/Workspace/corpus/"

utterance_tier = 1

Create Strings as file list: "FileList", "'input_dir$'*.TextGrid"
number_of_files = Get number of strings
file_counter = 0
for file_idx from 1 to number_of_files
	selectObject: "Strings FileList"
	file$ = Get string: file_idx
	Read from file: "'input_dir$''file$'"
	name$ = selected$("TextGrid")
	Read from file: "'input_dir$''name$'.wav"

	selectObject: "TextGrid 'name$'"
	number_of_intervals = Get number of intervals: utterance_tier

	for interval_idx from 1 to number_of_intervals
		selectObject: "TextGrid 'name$'"
		label$ = Get label of interval: utterance_tier, interval_idx

		if label$ <> "silent" and label$ <> "sounding"
			file_counter = file_counter + 1
			new_name$ = "'name$'_'file_counter'"
			tmin = Get start time of interval: utterance_tier, interval_idx
			tmax = Get end time of interval: utterance_tier, interval_idx
			
			selectObject: "Sound 'name$'"
			Extract part: tmin, tmax, "rectangular", 1, "no"
			Rename: "'new_name$'"
			Save as WAV file: "'output_dir$''new_name$'.wav"

			selectObject: "TextGrid 'name$'"
			Extract one tier: 1
			Rename: "'name$'_tier1"
			Extract part: tmin, tmax, "no"
			Rename: "'new_name$'"

			Save as text file: "'output_dir$''new_name$'.TextGrid"

			removeObject: "Sound 'new_name$'", "TextGrid 'new_name$'", "TextGrid 'name$'_tier1"
			

		endif

	endfor
	
	removeObject: "Sound 'name$'", "TextGrid 'name$'"
endfor
removeObject: "Strings FileList"
printline done! 'file_counter' utterances extracted



