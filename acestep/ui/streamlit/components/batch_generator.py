"""
Batch Generator component - generate multiple songs at once
"""
import streamlit as st
import pandas as pd
from utils import ProjectManager, get_dit_handler, get_llm_handler, is_dit_ready, is_llm_ready
from config import PROJECTS_DIR, DEFAULT_DURATION, DEFAULT_BPM
from loguru import logger
from acestep.inference import generate_music, GenerationParams, GenerationConfig
import os


def show_batch_generator():
    """Display batch generation interface (up to 8 songs)"""
    st.markdown("## 📦 Batch Generator")
    st.info("🚀 Generate up to 8 songs simultaneously")
    
    # Check models
    if not is_dit_ready():
        st.error("❌ DiT model not loaded. Please go to Settings → Models to initialize.")
        return
    
    # Initialize batch queue
    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []
    
    st.markdown("### Add Songs to Queue")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        song_caption = st.text_input(
            "Song Description",
            placeholder="Upbeat pop with synth...",
            key="batch_caption"
        )
    
    with col2:
        if st.button("➕ Add to Queue", key="batch_add_btn", use_container_width=True):
            if song_caption and len(st.session_state.batch_queue) < 8:
                st.session_state.batch_queue.append({
                    "caption": song_caption,
                    "duration": DEFAULT_DURATION,
                    "bpm": DEFAULT_BPM,
                    "status": "queued"
                })
                st.success("✅ Added to queue")
                st.rerun()
            elif len(st.session_state.batch_queue) >= 8:
                st.error("🔴 Queue is full (max 8 songs)")
            else:
                st.error("Please enter a song description")
    
    st.divider()
    
    # Queue display
    st.markdown(f"### Queue ({len(st.session_state.batch_queue)}/8)")
    
    if st.session_state.batch_queue:
        # Show as grid
        cols = st.columns(4)
        
        for idx, song in enumerate(st.session_state.batch_queue):
            with cols[idx % 4]:
                with st.container(border=True):
                    st.markdown(f"**#{idx + 1}**")
                    st.caption(song["caption"][:50] + "..." if len(song["caption"]) > 50 else song["caption"])
                    
                    # Status indicator
                    status_emoji = {
                        "queued": "⏳",
                        "generating": "⚙️",
                        "completed": "✅",
                        "failed": "❌"
                    }
                    st.caption(f"{status_emoji.get(song['status'], '?')} {song['status'].title()}")
                    
                    # Remove button
                    if song["status"] == "queued":
                        if st.button("🗑️", key=f"remove_{idx}", use_container_width=True):
                            st.session_state.batch_queue.pop(idx)
                            st.rerun()
    else:
        st.info("📝 Add songs to the queue to get started")
    
    st.divider()
    
    # Batch settings
    with st.expander("⚙️ Batch Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            inference_steps = st.slider(
                "Diffusion Steps",
                min_value=8,
                max_value=100,
                value=32,
                step=4,
                key="batch_steps"
            )
        
        with col2:
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=15.0,
                value=7.5,
                step=0.5,
                key="batch_guidance"
            )
        
        with col3:
            batch_size = st.slider(
                "Batch Size",
                min_value=1,
                max_value=4,
                value=1,
                help="Generate multiple variations per song",
                key="batch_size"
            )
        
        # Advanced settings
        col4, col5 = st.columns(2)
        with col4:
            use_cot = st.checkbox("Use Chain-of-Thought (LLM)", value=is_llm_ready(), key="batch_cot")
        with col5:
            audio_format = st.selectbox(
                "Output Format",
                ["wav32", "flac", "mp3"],
                index=0,
                key="batch_format"
            )
    
    st.divider()
    
    # Generate Button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(
            f"🚀 Generate All ({len(st.session_state.batch_queue)})",
            use_container_width=True,
            type="primary",
            key="batch_gen_btn",
            disabled=len(st.session_state.batch_queue) == 0
        ):
            generate_batch(
                st.session_state.batch_queue,
                inference_steps,
                guidance_scale,
                batch_size,
                use_cot,
                audio_format
            )


def generate_batch(queue: list, steps: int, guidance: float, batch_size: int, use_cot: bool, audio_format: str):
    """Generate all songs in the batch queue"""
    pm = ProjectManager(PROJECTS_DIR)
    dit_handler = get_dit_handler()
    llm_handler = get_llm_handler() if use_cot else None
    
    if not dit_handler:
        st.error("❌ Failed to load generation model")
        return
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_songs = len(queue)
    completed = 0
    failed = 0
    results = []
    
    # Create output directory
    output_dir = os.path.join(PROJECTS_DIR, "batch_output")
    os.makedirs(output_dir, exist_ok=True)
    
    with results_container:
        st.markdown("### 🎵 Generation Results")
        results_area = st.empty()
    
    for idx, song in enumerate(queue):
        # Update status
        song["status"] = "generating"
        progress = idx / total_songs
        progress_bar.progress(progress)
        status_text.text(f"🎵 Generating song {idx + 1}/{total_songs}: {song['caption'][:40]}...")
        
        try:
            # Create project
            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in song['caption'][:30])
            project_name = f"Batch_{idx+1:02d}_{safe_name.replace(' ', '_')}"
            project_path = pm.create_project(project_name, description=song['caption'])
            
            # Setup generation parameters
            params = GenerationParams(
                task_type="text2music",
                caption=song['caption'],
                lyrics="[Instrumental]",  # Default to instrumental for batch
                duration=song['duration'],
                bpm=song['bpm'] if song['bpm'] > 0 else None,
                inference_steps=steps,
                guidance_scale=guidance,
                seed=-1,  # Random seed
                thinking=use_cot and llm_handler is not None,
                use_cot_metas=use_cot,
                use_cot_caption=use_cot,
            )
            
            config = GenerationConfig(
                batch_size=batch_size,
                use_random_seed=True,
                audio_format=audio_format,
            )
            
            # Generate with progress
            def progress_callback(step, total_steps):
                inner_progress = (step / total_steps) * (1 / total_songs)
                progress_bar.progress(progress + inner_progress)
            
            # Run generation
            result = generate_music(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                params=params,
                config=config,
                save_dir=project_path,
                progress=progress_callback if batch_size == 1 else None,
            )
            
            if result.success:
                # Save metadata
                for audio in result.audios:
                    if audio['path']:
                        pm.save_audio_metadata(
                            project_path,
                            os.path.basename(audio['path']),
                            {
                                "caption": song['caption'],
                                "duration": song['duration'],
                                "bpm": song['bpm'],
                                "seed": audio['params'].get('seed'),
                                "status_message": result.status_message,
                            }
                        )
                
                results.append({
                    "song": song['caption'][:50],
                    "project": project_name,
                    "files": len(result.audios),
                    "status": "✅ Success"
                })
                song["status"] = "completed"
                completed += 1
            else:
                raise Exception(result.error or "Generation failed")
        
        except Exception as e:
            logger.error(f"Batch generation error for song {idx + 1}: {e}")
            results.append({
                "song": song['caption'][:50],
                "project": project_name if 'project_name' in locals() else "",
                "files": 0,
                "status": f"❌ Failed: {str(e)[:50]}"
            })
            song["status"] = "failed"
            failed += 1
    
    # Final update
    progress_bar.progress(1.0)
    status_text.empty()
    
    # Display summary
    with results_container:
        st.success(f"🎉 Batch generation complete! ✅ {completed} | ❌ {failed} | Total: {total_songs}")
        
        # Show results table
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Clear queue button
        if st.button("🗑️ Clear Queue", key="clear_batch_queue"):
            st.session_state.batch_queue = []
            st.rerun()