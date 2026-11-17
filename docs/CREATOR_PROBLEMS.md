# Creator problems → what AudioLayer does about them

AudioLayer was scoped from the complaints people actually post about the audio that comes out of
AI video generators (Veo 3.x, Sora 2, Kling 2.x/3.x, Wan, Runway…). Each row says what the tool
does, and how honest the fix is.

| Complaint (paraphrased from creator reports, reviews and forums) | What AudioLayer does | How well it works |
|---|---|---|
| "I asked for no music and it still added a soundtrack." | **Remove all music** preset removes BGM + foreground Music; per-segment mute for only part of it. | Real neural separation (Demucs). Vocals vs. everything else is strong; music vs. effects inside the non-vocal part is weaker (see Limitations). |
| "Characters babble in a made-up language / gibberish speech" (Kling, Veo, Sora inventing background chatter). | Whisper-based **intelligibility score** per dialogue segment; segments flagged `unintelligible`, `no_words_recognized`, `language_uncertain`. **Mute gibberish dialogue** preset mutes only those. | A hint, not a verdict: heavy accents or mumbling can also score low. You review the flagged segments. |
| "I want to replace the AI voice / dub it, but keep the sound design." | **Music & Effects (M&E)** preset = everything except dialogue, the standard deliverable for dubbing. Dialogue stem + SRT for re-voicing. | Good when voices are clearly separable; sung vocals and shouted voices inside music can leave residue. |
| "Audio sounds muffled / flat / low quality." | **Band-limited audio** detector measures the effective bandwidth (many generators synthesise at 16–24 kHz and upsample). | Detection + explanation only — separation cannot restore missing highs. |
| "The video came out silent" (no audio stream, or a stream full of digital silence; also after some upscalers). | Detects **no audio track** and **silent audio** up-front and says so instead of producing empty stems. | Reliable. |
| "Random sound effects that don't fit." | SFX lane with AudioSet labels (e.g. *Whoosh*, *Bang*, *Bird*); remove all SFX or mute single events. | Detection is good for clearly non-musical events; percussive/tonal effects can be heard as music by the classifier. |
| "Stitched 8–10 s clips jump in volume and click at the joins." | **Loudness presets** (YouTube/TikTok −14 LUFS, podcast −16, EBU R128 −23) via two-pass `loudnorm`; **edge fades** on export; edge-click detector. Normalising every clip to the same target makes them match. | Reliable (measured LUFS is reported after export). |
| "Hiss / hum / rumble in the generated audio." | Measured hiss, mains hum (50/60 Hz + harmonics), rumble, DC offset → Noise stem built only for what was found (high-pass, notch filters, spectral gating from the quietest frames). | Good for stationary noise. Non-stationary noise (wind gusts) is detected but not removed. |
| "Clipped / distorted audio." | Clipping regions marked on the Noise lane. | Detection only — clipping is baked into the waveform. |
| "Mono playback cancels the sound." | Stereo phase-correlation check; dual-mono detection. | Detection only. |
| "I need the parts in my editor, in sync." | Full-length stems (silence preserved, sample-accurate), cleaned mix, original audio, video remux with the video stream copied, ZIP with JSON/CSV/SRT and **Audacity label track**. | Reliable; verified by tests (frame counts equal the master). |
| "BGM vs. a song in the scene — the tool can't know." | Explicit BGM/Music rule (dialogue overlap, singing, level, AudioSet "background/soundtrack" hints) shown in the description; `role_uncertain` flag; one-click relabel re-routes the stems without re-running models. | Transparent rule, user-correctable. |

Sources used while scoping (Nov 2026): reports of Veo 3 producing silent clips and unwanted
subtitles; Kling's audio producing unintelligible, tonal speech; reviews describing Sora 2 audio as
muffled/flat and inventing background chatter; guides recommending stitching short clips; creator
wish-lists asking for "no unwanted background music" and better mixing.
