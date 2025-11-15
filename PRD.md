# Agentic sound editing pipeline

**Project summary**
From video or audio input, 
edit the audio with simple additional sounds (dog barking, humoristic sound, ...)
to have a dynamic and stimulating rendering. 

Input: 
- video or audio
- (optional : prompt user input)

Steps :
- (extract audio only from video)
- speech-to-text : recover the narrated text, as natural language
- word-embedding, tokenization of the speech
- word-embedding, tokenization of sounds metadata (a title and a short description)
- vector-matching between speech and sounds 
- filtering and selection of sounds and timings
- audio editing : add sounds to the original audio track

Guidelines :
- each step can have input-output dump folders for now
- idea to structure each as a independant micro-service
- audio files as .wav
