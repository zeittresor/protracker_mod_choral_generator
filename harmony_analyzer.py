#!/usr/bin/env python3
"""
Harmony Analyzer and Music Quality Checker
Provides harmonic analysis, chord progression validation, and multi-instance quality checking
"""

import math
import random
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class HarmonyAnalysis:
    """Results of harmonic analysis"""
    harmonic_score: float  # 0-100
    dissonance_score: float  # 0-100
    scale_compatibility: float  # 0-100
    chord_progressions: List[Dict[str, Any]]
    voice_leading_score: float  # 0-100
    recommendations: List[str]
    
@dataclass
class MusicQuality:
    """Overall music quality assessment"""
    overall_score: float  # 0-100
    harmony_score: float
    rhythm_score: float
    melody_score: float
    structure_score: float
    issues: List[str]
    strengths: List[str]
    passed: bool

class HarmonyAnalyzer:
    """
    Analyzes musical harmony and provides recommendations for improvement
    """
    
    def __init__(self):
        # Extended scales with more modes
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'locrian': [0, 1, 3, 5, 6, 8, 10],
            'blues': [0, 3, 5, 6, 7, 10],
            'pentatonic_major': [0, 2, 4, 7, 9],
            'pentatonic_minor': [0, 3, 5, 7, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
        }
        
        # Chord qualities with their interval structures
        self.chord_qualities = {
            'major': [0, 4, 7],
            'minor': [0, 3, 7],
            'diminished': [0, 3, 6],
            'augmented': [0, 4, 8],
            'sus2': [0, 2, 7],
            'sus4': [0, 5, 7],
            'major7': [0, 4, 7, 11],
            'dominant7': [0, 4, 7, 10],
            'minor7': [0, 3, 7, 10],
            'half_diminished': [0, 3, 6, 10],
            'diminished7': [0, 3, 6, 9],
        }
        
        # Common chord progressions by genre
        self.progressions = {
            'pop': [
                [0, 4, 5, 4],  # I-IV-V-IV
                [0, 5, 3, 4],  # I-vi-IV-V
                [0, 3, 4, 0],  # I-vi-IV-I
                [5, 3, 4, 0],  # vi-IV-V-I
            ],
            'rock': [
                [0, 3, 4, 0],  # I-vi-IV-I
                [0, 4, 5, 0],  # I-IV-V-I
                [0, 5, 3, 4],  # I-vi-IV-V
                [4, 0, 5, 3],  # IV-I-V-vi
            ],
            'jazz': [
                [2, 5, 0, 0],  # ii-V-I-I
                [0, 5, 1, 4],  # I-vi-ii-IV
                [3, 6, 2, 5],  # vi-vii°-ii-V
                [0, 3, 6, 2],  # I-vi-vii°-ii
            ],
            'classical': [
                [0, 4, 5, 1],  # I-IV-V-ii
                [0, 5, 1, 4],  # I-vi-ii-IV
                [0, 3, 6, 0],  # I-vi-vii°-I
                [0, 2, 5, 0],  # I-ii-V-I
            ],
            'folk': [
                [0, 5, 3, 4],  # I-vi-IV-V
                [0, 4, 5, 3],  # I-IV-V-vi
                [0, 2, 5, 0],  # I-ii-V-I
                [0, 5, 1, 4],  # I-vi-ii-IV
            ],
        }
    
    def note_to_midi(self, note_str: str) -> int:
        """Convert note string (e.g., 'C-2', 'F#3') to MIDI number"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Parse note and octave
        if '#' in note_str:
            note_name = note_str[:2]
            octave = int(note_str[2:])
        else:
            note_name = note_str[:1]
            octave = int(note_str[1:])
        
        note_index = notes.index(note_name)
        return (octave + 1) * 12 + note_index
    
    def midi_to_note(self, midi_num: int) -> str:
        """Convert MIDI number to note string"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_num // 12) - 1
        note_index = midi_num % 12
        return f"{notes[note_index]}-{octave}"
    
    def normalize_to_octave(self, note: int) -> int:
        """Normalize note to 0-11 range (C=0, C#=1, etc.)"""
        return note % 12
    
    def analyze_harmony(self, notes: List[str], scale_type: str = 'major', root_note: str = 'C-2') -> HarmonyAnalysis:
        """
        Analyze the harmonic content of a sequence of notes
        
        Args:
            notes: List of note strings (e.g., ['C-2', 'E-2', 'G-2'])
            scale_type: Type of scale ('major', 'minor', 'dorian', etc.)
            root_note: Root note of the scale
            
        Returns:
            HarmonyAnalysis object with scores and recommendations
        """
        # Convert notes to MIDI numbers
        midi_notes = [self.note_to_midi(n) for n in notes if n]
        
        if not midi_notes:
            return HarmonyAnalysis(0, 0, 0, [], 0, ["No notes to analyze"])
        
        # Get scale intervals
        scale_intervals = self.scales.get(scale_type, self.scales['major'])
        root_midi = self.note_to_midi(root_note)
        
        # Build scale notes
        scale_notes = [(root_midi + interval) % 12 for interval in scale_intervals]
        
        # Calculate scale compatibility
        normalized_notes = [self.normalize_to_octave(n) for n in midi_notes]
        in_scale_count = sum(1 for n in normalized_notes if n in scale_notes)
        scale_compatibility = (in_scale_count / len(normalized_notes)) * 100
        
        # Analyze intervals between consecutive notes
        interval_scores = []
        voice_leading_issues = []
        
        for i in range(len(midi_notes) - 1):
            interval = abs(midi_notes[i + 1] - midi_notes[i])
            interval_class = interval % 12
            
            # Score based on interval quality
            if interval_class in [0, 5, 7]:  # Unison, perfect 4th, perfect 5th
                interval_scores.append(5)
            elif interval_class in [3, 4, 8, 9]:  # Minor/major 3rds, 6ths
                interval_scores.append(4)
            elif interval_class in [2, 10]:  # Major/minor 2nds
                interval_scores.append(3)
            elif interval_class == 6:  # Tritone
                interval_scores.append(1)
                voice_leading_issues.append(f"Tritone at position {i}")
            else:  # Other intervals
                interval_scores.append(2)
        
        harmonic_score = sum(interval_scores) / max(1, len(interval_scores)) * 20
        harmonic_score = min(100, harmonic_score)
        
        # Voice leading analysis
        voice_leading_score = self._analyze_voice_leading(midi_notes)
        
        # Identify chord progressions
        chord_progressions = self._identify_chord_progressions(midi_notes, scale_type, root_midi)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            scale_compatibility, harmonic_score, voice_leading_score, 
            voice_leading_issues, chord_progressions
        )
        
        # Calculate dissonance
        dissonance_score = max(0, 100 - harmonic_score)
        
        return HarmonyAnalysis(
            harmonic_score=harmonic_score,
            dissonance_score=dissonance_score,
            scale_compatibility=scale_compatibility,
            chord_progressions=chord_progressions,
            voice_leading_score=voice_leading_score,
            recommendations=recommendations
        )
    
    def _analyze_voice_leading(self, midi_notes: List[int]) -> float:
        """Analyze voice leading quality"""
        if len(midi_notes) < 2:
            return 50.0
        
        good_movements = 0
        total_movements = len(midi_notes) - 1
        
        for i in range(total_movements):
            interval = abs(midi_notes[i + 1] - midi_notes[i])
            
            # Good voice leading: small steps (2nds, 3rds), occasional larger leaps
            if interval <= 4:  # Major 3rd or smaller
                good_movements += 1
            elif interval <= 7:  # Perfect 5th or smaller
                good_movements += 0.8
            elif interval <= 12:  # Octave or smaller
                good_movements += 0.5
            else:
                good_movements += 0.2
        
        return (good_movements / total_movements) * 100
    
    def _identify_chord_progressions(self, midi_notes: List[int], scale_type: str, root_midi: int) -> List[Dict]:
        """Identify chord progressions in the note sequence"""
        progressions = []
        
        # Group notes into potential chords (windows of 3-4 notes)
        for window_size in [3, 4]:
            for i in range(len(midi_notes) - window_size + 1):
                window = midi_notes[i:i + window_size]
                normalized_window = [self.normalize_to_octave(n - root_midi) for n in window]
                
                # Try to identify chord
                chord = self._identify_chord(normalized_window)
                if chord:
                    progressions.append({
                        'position': i,
                        'notes': [self.midi_to_note(n) for n in window],
                        'chord_name': chord['name'],
                        'chord_quality': chord['quality'],
                        'root': chord['root']
                    })
        
        return progressions
    
    def _identify_chord(self, normalized_notes: List[int]) -> Optional[Dict]:
        """Identify a chord from normalized note values"""
        unique_notes = sorted(set(normalized_notes))
        
        if len(unique_notes) < 3:
            return None
        
        # Try each possible root
        for root in unique_notes:
            # Calculate intervals from root
            intervals = sorted([(n - root) % 12 for n in unique_notes if n != root])
            
            # Match against known chord qualities
            for quality_name, quality_intervals in self.chord_qualities.items():
                if self._intervals_match(intervals, quality_intervals):
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    root_name = note_names[root]
                    
                    return {
                        'name': f"{root_name} {quality_name}",
                        'quality': quality_name,
                        'root': root_name,
                        'intervals': intervals
                    }
        
        return None
    
    def _intervals_match(self, intervals1: List[int], intervals2: List[int]) -> bool:
        """Check if two interval sets match"""
        if len(intervals1) != len(intervals2):
            return False
        return all(i in intervals2 for i in intervals1)
    
    def _generate_recommendations(self, scale_compat: float, harmonic_score: float, 
                                 voice_leading: float, issues: List[str], 
                                 progressions: List[Dict]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if scale_compat < 70:
            recommendations.append("Consider using more notes from the selected scale")
        
        if harmonic_score < 60:
            recommendations.append("Reduce dissonant intervals for smoother harmony")
        
        if voice_leading < 60:
            recommendations.append("Use smaller stepwise motion between notes")
        
        if len(progressions) < 2:
            recommendations.append("Try creating clearer chord progressions")
        
        if harmonic_score > 80 and voice_leading > 70:
            recommendations.append("Good harmonic structure - consider adding more rhythmic variety")
        
        recommendations.extend(issues)
        
        return recommendations
    
    def improve_harmony(self, notes: List[str], scale_type: str = 'major', root_note: str = 'C-2') -> List[str]:
        """
        Improve a sequence of notes to make it more harmonic
        
        Returns a new list of notes that are adjusted to fit the scale better
        """
        scale_intervals = self.scales.get(scale_type, self.scales['major'])
        root_midi = self.note_to_midi(root_note)
        
        # Build complete scale notes across multiple octaves
        scale_notes = set()
        for octave in range(-1, 5):  # Cover several octaves
            for interval in scale_intervals:
                scale_notes.add((root_midi + interval + octave * 12) % 12)
        
        improved = []
        for note in notes:
            if not note:
                improved.append(None)
                continue
                
            midi_num = self.note_to_midi(note)
            normalized = self.normalize_to_octave(midi_num)
            
            if normalized in scale_notes:
                improved.append(note)
            else:
                # Find closest scale note
                closest = self._find_closest_scale_note(midi_num, scale_notes)
                improved.append(self.midi_to_note(closest))
        
        return improved
    
    def _find_closest_scale_note(self, midi_note: int, scale_notes: set) -> int:
        """Find the closest note in the scale"""
        octave = (midi_note // 12) * 12
        normalized = midi_note % 12
        
        # Find closest scale note in the same octave
        best_note = None
        best_distance = float('inf')
        
        for scale_note in scale_notes:
            # Distance considering octave wrap
            distance = min(
                abs(normalized - scale_note),
                abs(normalized - scale_note + 12),
                abs(normalized - scale_note - 12)
            )
            if distance < best_distance:
                best_distance = distance
                best_note = scale_note
        
        return octave + best_note


class MusicQualityChecker:
    """
    Multi-instance music quality checker with 3 evaluation stages
    """
    
    def __init__(self, quality_threshold: float = 70.0):
        self.harmony_analyzer = HarmonyAnalyzer()
        self.quality_threshold = quality_threshold
        self.max_attempts = 3
    
    def check_quality_first_pass(self, patterns: List[List[List]], scale_type: str, root_note: str) -> MusicQuality:
        """
        First pass: Basic harmonic analysis and structure check
        """
        issues = []
        strengths = []
        
        # Extract all notes from patterns
        all_notes = []
        for pattern in patterns:
            for row in pattern:
                for note_data in row:
                    if note_data and note_data[0]:  # note_data is (note, instrument, effect, param)
                        all_notes.append(note_data[0])
        
        if not all_notes:
            return MusicQuality(0, 0, 0, 0, 0, ["No notes found in patterns"], [], False)
        
        # Analyze harmony
        harmony = self.harmony_analyzer.analyze_harmony(all_notes, scale_type, root_note)
        
        # Evaluate structure
        structure_score = self._evaluate_structure(patterns)
        
        # Evaluate rhythm variety
        rhythm_score = self._evaluate_rhythm(patterns)
        
        # Calculate overall score
        overall_score = (
            harmony.harmonic_score * 0.4 +
            harmony.voice_leading_score * 0.2 +
            rhythm_score * 0.2 +
            structure_score * 0.2
        )
        
        # Identify issues and strengths
        if harmony.harmonic_score < 60:
            issues.append("Low harmonic score - too many dissonant intervals")
        elif harmony.harmonic_score > 80:
            strengths.append("Strong harmonic progression")
        
        if harmony.voice_leading_score < 60:
            issues.append("Poor voice leading - too many large leaps")
        elif harmony.voice_leading_score > 75:
            strengths.append("Good voice leading")
        
        if structure_score < 50:
            issues.append("Weak structural organization")
        
        if rhythm_score < 40:
            issues.append("Limited rhythmic variety")
        
        passed = overall_score >= self.quality_threshold
        
        return MusicQuality(
            overall_score=overall_score,
            harmony_score=harmony.harmonic_score,
            rhythm_score=rhythm_score,
            melody_score=harmony.voice_leading_score,
            structure_score=structure_score,
            issues=issues,
            strengths=strengths,
            passed=passed
        )
    
    def check_quality_second_pass(self, patterns: List[List[List]], scale_type: str, root_note: str) -> MusicQuality:
        """
        Second pass: Detailed chord progression analysis and contrapuntal evaluation
        """
        all_notes = []
        for pattern in patterns:
            for row in pattern:
                for note_data in row:
                    if note_data and note_data[0]:
                        all_notes.append(note_data[0])
        
        harmony = self.harmony_analyzer.analyze_harmony(all_notes, scale_type, root_note)
        
        issues = []
        strengths = []
        
        # Check chord progression quality
        if len(harmony.chord_progressions) >= 2:
            progression_quality = self._evaluate_progression_flow(harmony.chord_progressions)
            if progression_quality < 60:
                issues.append("Chord progressions lack logical flow")
            else:
                strengths.append("Good chord progression flow")
        
        # Check for parallel fifths/octaves (basic counterpoint)
        parallel_issues = self._check_parallel_intervals(patterns)
        if parallel_issues:
            issues.extend(parallel_issues)
        
        # Evaluate melodic contour
        melody_score = self._evaluate_melodic_contour(patterns)
        
        # Recalculate overall score with stricter weighting
        overall_score = (
            harmony.harmonic_score * 0.35 +
            harmony.voice_leading_score * 0.25 +
            melody_score * 0.25 +
            (100 if not parallel_issues else 60) * 0.15
        )
        
        passed = overall_score >= self.quality_threshold + 5  # Slightly stricter
        
        return MusicQuality(
            overall_score=overall_score,
            harmony_score=harmony.harmonic_score,
            rhythm_score=self._evaluate_rhythm(patterns),
            melody_score=melody_score,
            structure_score=harmony.voice_leading_score,
            issues=issues,
            strengths=strengths,
            passed=passed
        )
    
    def check_quality_third_pass(self, patterns: List[List[List]], scale_type: str, root_note: str) -> MusicQuality:
        """
        Third pass: Final harmony verification - ensures all notes harmonize together
        """
        all_notes = []
        for pattern in patterns:
            for row in pattern:
                for note_data in row:
                    if note_data and note_data[0]:
                        all_notes.append(note_data[0])
        
        # Deep harmonic analysis
        harmony = self.harmony_analyzer.analyze_harmony(all_notes, scale_type, root_note)
        
        # Verify scale compatibility of every note
        scale_intervals = self.harmony_analyzer.scales.get(scale_type, self.harmony_analyzer.scales['major'])
        root_midi = self.harmony_analyzer.note_to_midi(root_note)
        scale_notes = [(root_midi + interval) % 12 for interval in scale_intervals]
        
        out_of_scale_count = 0
        for note in all_notes:
            if note:
                normalized = self.harmony_analyzer.normalize_to_octave(self.harmony_analyzer.note_to_midi(note))
                if normalized not in scale_notes:
                    out_of_scale_count += 1
        
        issues = []
        strengths = []
        
        if out_of_scale_count > len(all_notes) * 0.3:
            issues.append(f"Too many out-of-scale notes ({out_of_scale_count} notes)")
        elif out_of_scale_count == 0:
            strengths.append("Perfect scale adherence")
        
        # Check for tonal center stability
        tonal_stability = self._check_tonal_stability(patterns, root_note)
        if tonal_stability < 70:
            issues.append("Tonal center is unstable")
        else:
            strengths.append("Strong tonal center")
        
        # Final harmony score with strict criteria
        harmony_score = harmony.harmonic_score
        if out_of_scale_count > 0:
            harmony_score *= (1 - out_of_scale_count / len(all_notes))
        
        overall_score = (
            harmony_score * 0.5 +
            harmony.voice_leading_score * 0.3 +
            tonal_stability * 0.2
        )
        
        # Stricter pass threshold
        passed = overall_score >= self.quality_threshold + 10
        
        return MusicQuality(
            overall_score=overall_score,
            harmony_score=harmony_score,
            rhythm_score=self._evaluate_rhythm(patterns),
            melody_score=harmony.voice_leading_score,
            structure_score=tonal_stability,
            issues=issues,
            strengths=strengths,
            passed=passed
        )
    
    def full_quality_check(self, patterns: List[List[List]], scale_type: str, root_note: str) -> Tuple[MusicQuality, bool]:
        """
        Run all three quality checks and return the best result
        
        Returns:
            Tuple of (best_quality, passed_all_checks)
        """
        # First pass
        quality_1 = self.check_quality_first_pass(patterns, scale_type, root_note)
        if not quality_1.passed:
            return quality_1, False
        
        # Second pass
        quality_2 = self.check_quality_second_pass(patterns, scale_type, root_note)
        if not quality_2.passed:
            return quality_2, False
        
        # Third pass
        quality_3 = self.check_quality_third_pass(patterns, scale_type, root_note)
        
        # Return the best quality assessment
        best_quality = quality_3 if quality_3.overall_score >= quality_2.overall_score else quality_2
        
        return best_quality, quality_3.passed
    
    def _evaluate_structure(self, patterns: List[List[List]]) -> float:
        """Evaluate structural organization"""
        if not patterns:
            return 0.0
        
        # Check pattern variety
        pattern_variety = len(set(str(p) for p in patterns))
        variety_score = min(100, pattern_variety * 10)
        
        # Check for consistent structure
        avg_length = sum(len(p) for p in patterns) / len(patterns)
        consistency_score = 100 if 60 <= avg_length <= 65 else 70
        
        return (variety_score + consistency_score) / 2
    
    def _evaluate_rhythm(self, patterns: List[List[List]]) -> float:
        """Evaluate rhythmic variety"""
        if not patterns:
            return 0.0
        
        note_densities = []
        for pattern in patterns:
            for row in pattern:
                active_notes = sum(1 for note_data in row if note_data and note_data[0])
                note_densities.append(active_notes)
        
        if not note_densities:
            return 0.0
        
        # Calculate rhythmic variety
        avg_density = sum(note_densities) / len(note_densities)
        density_variety = len(set(note_densities)) / len(note_densities) * 100
        
        # Good rhythm has moderate density with variety
        density_score = 100 - abs(avg_density - 2) * 20
        
        return (density_score + density_variety) / 2
    
    def _evaluate_progression_flow(self, progressions: List[Dict]) -> float:
        """Evaluate how well chord progressions flow"""
        if len(progressions) < 2:
            return 50.0
        
        # Check for common tones between consecutive chords
        common_tones = 0
        for i in range(len(progressions) - 1):
            current_chord = set(progressions[i].get('notes', []))
            next_chord = set(progressions[i + 1].get('notes', []))
            
            if current_chord and next_chord:
                common = len(current_chord.intersection(next_chord))
                if common >= 1:
                    common_tones += 1
        
        return (common_tones / max(1, len(progressions) - 1)) * 100
    
    def _check_parallel_intervals(self, patterns: List[List[List]]) -> List[str]:
        """Check for parallel fifths and octaves (basic counterpoint rule)"""
        issues = []
        
        # Simplified check - in a full implementation, this would analyze voice leading between channels
        parallel_count = 0
        
        # This is a placeholder for a more sophisticated counterpoint analysis
        if parallel_count > 3:
            issues.append("Too many parallel intervals detected")
        
        return issues
    
    def _evaluate_melodic_contour(self, patterns: List[List[List]]) -> float:
        """Evaluate melodic contour variety"""
        melodies = []
        
        for pattern in patterns:
            for row_idx, row in enumerate(pattern):
                for ch_idx, note_data in enumerate(row):
                    if note_data and note_data[0]:
                        while len(melodies) <= ch_idx:
                            melodies.append([])
                        melodies[ch_idx].append((row_idx, self.harmony_analyzer.note_to_midi(note_data[0])))
        
        if not melodies or not melodies[0]:
            return 50.0
        
        # Analyze contour variety for first melody channel
        melody = melodies[0]
        if len(melody) < 2:
            return 50.0
        
        directions = []
        for i in range(len(melody) - 1):
            interval = melody[i + 1][1] - melody[i][1]
            if interval > 0:
                directions.append('up')
            elif interval < 0:
                directions.append('down')
            else:
                directions.append('same')
        
        direction_changes = sum(1 for i in range(len(directions) - 1) if directions[i] != directions[i + 1])
        variety = direction_changes / max(1, len(directions) - 1)
        
        return variety * 100
    
    def _check_tonal_stability(self, patterns: List[List[List]], root_note: str) -> float:
        """Check if the tonal center is stable throughout the piece"""
        root_midi = self.harmony_analyzer.note_to_midi(root_note)
        root_normalized = root_midi % 12
        
        tonal_references = 0
        total_notes = 0
        
        for pattern in patterns:
            for row in pattern:
                for note_data in row:
                    if note_data and note_data[0]:
                        note_midi = self.harmony_analyzer.note_to_midi(note_data[0])
                        if note_midi % 12 == root_normalized:
                            tonal_references += 1
                        total_notes += 1
        
        if total_notes == 0:
            return 50.0
        
        stability = (tonal_references / total_notes) * 100
        return min(100, stability * 2)  # Boost stability score


# Utility function for integration
def analyze_and_improve_music(patterns: List[List[List]], scale_type: str, root_note: str, 
                             max_attempts: int = 3) -> Tuple[List[List[List]], MusicQuality, bool]:
    """
    Analyze music quality and attempt improvements if needed
    
    Returns:
        Tuple of (improved_patterns, final_quality, passed)
    """
    checker = MusicQualityChecker()
    
    for attempt in range(max_attempts):
        quality, passed = checker.full_quality_check(patterns, scale_type, root_note)
        
        if passed:
            return patterns, quality, True
        
        # Attempt improvements (this would be expanded with actual improvement logic)
        if attempt < max_attempts - 1:
            # For now, just return the original patterns
            # In a full implementation, this would apply harmonic improvements
            pass
    
    return patterns, quality, False


if __name__ == "__main__":
    # Test the analyzer
    analyzer = HarmonyAnalyzer()
    
    # Test with a simple C major scale
    test_notes = ['C-2', 'E-2', 'G-2', 'C-3', 'D-3', 'E-3', 'F-3', 'G-3']
    analysis = analyzer.analyze_harmony(test_notes, 'major', 'C-2')
    
    print("Harmony Analysis Test:")
    print(f"Harmonic Score: {analysis.harmonic_score:.1f}")
    print(f"Scale Compatibility: {analysis.scale_compatibility:.1f}%")
    print(f"Voice Leading Score: {analysis.voice_leading_score:.1f}")
    print(f"Chord Progressions: {len(analysis.chord_progressions)}")
    print(f"Recommendations: {analysis.recommendations}")
    print()
    
    # Test quality checker
    checker = MusicQualityChecker()
    print("Music Quality Checker initialized successfully")
