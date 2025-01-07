import pyaudio
import numpy as np
import wave
import time
import webrtcvad
import scipy.signal as signal

# Constants
CHUNK = 960  # 10ms frame size for 48000 Hz sample rate
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
NOISE_PROFILE_DURATION = 10  # seconds
OUTPUT_FILENAME = "cleaned_audio.wav"
MICROPHONE_INDEX = None  # Use default microphone
LATENCY_TARGET = 100  # in milliseconds
DYNAMIC_NOISE_ALPHA = 0.1  # Weight for dynamic noise updating

# VAD Settings
vad = webrtcvad.Vad()
vad.set_mode(2)  # 0 = most aggressive, 3 = least aggressive

# Filter parameters
CUTOFF_FREQUENCY_LOW = 200  # Hz (for high-pass filter)
CUTOFF_FREQUENCY_HIGH = 2000  # Hz (for low-pass filter)
FILTER_ORDER = 5
nyquist = 0.5 * RATE
normal_cutoff_high = CUTOFF_FREQUENCY_HIGH / nyquist
normal_cutoff_low = CUTOFF_FREQUENCY_LOW / nyquist
b_high, a_high = signal.butter(FILTER_ORDER, normal_cutoff_low, btype='high', analog=False)
b_low, a_low = signal.butter(FILTER_ORDER, normal_cutoff_high, btype='low', analog=False)

# Helper Functions
def perform_spectral_subtraction(noisy_frame, noise_profile):
    """
    Perform spectral subtraction to reduce noise.
    """
    noisy_spectrum = np.fft.rfft(noisy_frame)
    noise_spectrum = np.fft.rfft(noise_profile)
    gain = np.maximum(np.abs(noisy_spectrum) - np.abs(noise_spectrum), 0)
    clean_spectrum = gain * np.exp(1j * np.angle(noisy_spectrum))
    clean_frame = np.fft.irfft(clean_spectrum)
    return clean_frame

def update_noise_profile(noise_profile, new_chunk, speech_detected):
    """
    Dynamically update the noise profile.
    """
    if not speech_detected:
        return (1 - DYNAMIC_NOISE_ALPHA) * noise_profile + DYNAMIC_NOISE_ALPHA * new_chunk
    return noise_profile

def is_speech_chunk(chunk):
    """
    Determine if the audio chunk contains speech using VAD.
    """
    try:
        return vad.is_speech(chunk.tobytes(), RATE)
    except ValueError:
        return False

def apply_filters(clean_chunk):
    """
    Apply high-pass and low-pass filters to clean the audio further.
    """
    filtered_chunk = signal.lfilter(b_high, a_high, clean_chunk)  # High-pass filter
    filtered_chunk = signal.lfilter(b_low, a_low, filtered_chunk)  # Low-pass filter
    return filtered_chunk

def safe_read(stream, chunk_size):
    """
    Safely read from the audio stream to handle disconnections or errors.
    """
    try:
        return stream.read(chunk_size, exception_on_overflow=False)
    except (OSError, IOError) as e:
        print(f"Stream read error: {e}")
        return None

def safe_open_stream(p, **kwargs):
    """
    Safely open an audio stream to handle invalid device indices or other errors.
    """
    try:
        return p.open(**kwargs)
    except (OSError, IOError) as e:
        print(f"Error opening stream: {e}")
        return None

# Noise Canceller Class
class NoiseCanceller:
    def __init__(self, sample_rate, chunk_size):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_profile = np.zeros(chunk_size)

    def collect_initial_noise_profile(self, stream):
        """
        Collect an initial noise profile.
        """
        print("Collecting initial noise profile...")
        noise_profile_samples = []
        for _ in range(int(NOISE_PROFILE_DURATION * self.sample_rate / self.chunk_size)):
            audio_data = safe_read(stream, self.chunk_size)
            if audio_data is None:
                continue
            chunk = np.frombuffer(audio_data, dtype=np.int16)
            noise_profile_samples.append(chunk)
        if noise_profile_samples:
            self.noise_profile = np.mean(noise_profile_samples, axis=0)

    def process_audio(self, audio_data):
        """
        Process an audio chunk for noise cancellation.
        """
        chunk = np.frombuffer(audio_data, dtype=np.int16)
        speech_detected = is_speech_chunk(chunk)
        if speech_detected:
            clean_chunk = perform_spectral_subtraction(chunk, self.noise_profile)
            clean_chunk = apply_filters(clean_chunk)
            clean_chunk = np.clip(clean_chunk, -2**15, 2**15 - 1).astype(np.int16)
        else:
            clean_chunk = chunk
            self.noise_profile = update_noise_profile(self.noise_profile, chunk, speech_detected)
        return clean_chunk

def main():
    """
    Main function to handle real-time noise cancellation.
    """
    p = pyaudio.PyAudio()

    try:
        # Open microphone stream
        stream = safe_open_stream(
            p,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=MICROPHONE_INDEX,
            frames_per_buffer=CHUNK
        )

        if stream is None:
            print("Failed to open input stream.")
            return

        # Open output stream
        output_stream = safe_open_stream(
            p,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True
        )

        if output_stream is None:
            print("Failed to open output stream.")
            return

        # Open output file
        try:
            output_wave = wave.open(OUTPUT_FILENAME, 'wb')
            output_wave.setnchannels(CHANNELS)
            output_wave.setsampwidth(p.get_sample_size(FORMAT))
            output_wave.setframerate(RATE)
        except (OSError, IOError) as e:
            print(f"Error opening output file: {e}")
            return

        # Initialize noise canceller
        noise_canceller = NoiseCanceller(sample_rate=RATE, chunk_size=CHUNK)
        noise_canceller.collect_initial_noise_profile(stream)

        print("Starting real-time noise cancellation...")
        latency_measurements = []
        chunk_counter = 0

        while True:
            start_time = time.time()

            # Read audio chunk
            audio_data = safe_read(stream, CHUNK)
            if audio_data is None:
                continue

            # Process audio chunk
            cleaned_chunk = noise_canceller.process_audio(audio_data)

            # Write to output stream
            output_stream.write(cleaned_chunk.tobytes())

            # Save to file
            output_wave.writeframes(cleaned_chunk.tobytes())

            # Measure latency
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            latency_measurements.append(latency)

            if chunk_counter % 20 == 0:
                avg_latency = sum(latency_measurements) / len(latency_measurements)
                print(f"Average Latency: {avg_latency:.2f} ms")
                if avg_latency > LATENCY_TARGET:
                    print(f"Warning: Latency exceeds target ({LATENCY_TARGET} ms).")
                latency_measurements = []

            chunk_counter += 1

    except KeyboardInterrupt:
        print("\nStopping noise cancellation...")

    finally:
        # Cleanup
        if stream:
            stream.stop_stream()
            stream.close()
        if output_stream:
            output_stream.stop_stream()
            output_stream.close()
        if 'output_wave' in locals():
            output_wave.close()
        p.terminate()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()
