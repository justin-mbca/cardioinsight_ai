"""
CardioInsight AI - Teaching Module

This module provides tools for AI-assisted teaching of ECG interpretation.
It includes case generation, quiz functionality, and performance tracking.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import random
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

class CaseLibrary:
    """
    Class for managing a library of ECG cases for teaching.
    """
    
    def __init__(self, library_path=None):
        """
        Initialize the case library.
        
        Parameters:
        -----------
        library_path : str or None
            Path to the case library. If None, creates a new library.
        """
        self.cases = []
        self.library_path = library_path
        
        if library_path is not None and os.path.exists(library_path):
            self.load_library(library_path)
            
    def add_case(self, ecg_data, diagnosis, difficulty='medium', metadata=None):
        """
        Add a case to the library.
        
        Parameters:
        -----------
        ecg_data : array-like or str
            ECG data or path to ECG data file.
        diagnosis : str
            Correct diagnosis for the case.
        difficulty : str
            Difficulty level. Options: 'easy', 'medium', 'hard'. Default is 'medium'.
        metadata : dict or None
            Additional metadata for the case. If None, uses empty dict.
            
        Returns:
        --------
        case_id : str
            ID of the added case.
        """
        # Generate case ID
        case_id = f"case_{len(self.cases) + 1:04d}"
        
        # Set metadata
        if metadata is None:
            metadata = {}
            
        metadata['added_date'] = datetime.now().strftime('%Y-%m-%d')
        metadata['difficulty'] = difficulty
        
        # Create case
        case = {
            'id': case_id,
            'ecg_data': ecg_data if isinstance(ecg_data, str) else None,
            'diagnosis': diagnosis,
            'metadata': metadata
        }
        
        # If ECG data is provided as array, save it to file
        if not isinstance(ecg_data, str):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
            
            # Save ECG data
            ecg_path = os.path.join(os.path.dirname(self.library_path), f"{case_id}.npy")
            np.save(ecg_path, ecg_data)
            case['ecg_data'] = ecg_path
            
        # Add case to library
        self.cases.append(case)
        
        # Save library
        if self.library_path is not None:
            self.save_library()
            
        return case_id
    
    def get_case(self, case_id):
        """
        Get a case from the library.
        
        Parameters:
        -----------
        case_id : str
            ID of the case.
            
        Returns:
        --------
        case : dict
            Case data.
        """
        # Find case
        for case in self.cases:
            if case['id'] == case_id:
                # Load ECG data if path is provided
                if isinstance(case['ecg_data'], str) and os.path.exists(case['ecg_data']):
                    ecg_data = np.load(case['ecg_data'])
                    
                    # Create a copy of the case with loaded ECG data
                    case_copy = case.copy()
                    case_copy['ecg_data'] = ecg_data
                    return case_copy
                
                return case
                
        raise ValueError(f"Case {case_id} not found in library.")
    
    def get_random_case(self, difficulty=None, diagnosis=None):
        """
        Get a random case from the library.
        
        Parameters:
        -----------
        difficulty : str or None
            Difficulty level. If None, selects from any difficulty.
        diagnosis : str or None
            Diagnosis. If None, selects from any diagnosis.
            
        Returns:
        --------
        case : dict
            Case data.
        """
        # Filter cases
        filtered_cases = self.cases
        
        if difficulty is not None:
            filtered_cases = [case for case in filtered_cases if case['metadata']['difficulty'] == difficulty]
            
        if diagnosis is not None:
            filtered_cases = [case for case in filtered_cases if case['diagnosis'] == diagnosis]
            
        if not filtered_cases:
            raise ValueError("No cases match the specified criteria.")
            
        # Select random case
        case = random.choice(filtered_cases)
        
        # Get case with loaded ECG data
        return self.get_case(case['id'])
    
    def get_cases_by_diagnosis(self, diagnosis):
        """
        Get all cases with a specific diagnosis.
        
        Parameters:
        -----------
        diagnosis : str
            Diagnosis to filter by.
            
        Returns:
        --------
        cases : list
            List of case IDs.
        """
        return [case['id'] for case in self.cases if case['diagnosis'] == diagnosis]
    
    def get_cases_by_difficulty(self, difficulty):
        """
        Get all cases with a specific difficulty level.
        
        Parameters:
        -----------
        difficulty : str
            Difficulty level to filter by.
            
        Returns:
        --------
        cases : list
            List of case IDs.
        """
        return [case['id'] for case in self.cases if case['metadata']['difficulty'] == difficulty]
    
    def save_library(self, path=None):
        """
        Save the case library.
        
        Parameters:
        -----------
        path : str or None
            Path to save the library. If None, uses the library_path.
        """
        if path is None:
            path = self.library_path
            
        if path is None:
            raise ValueError("No path specified for saving library.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save library
        with open(path, 'w') as f:
            json.dump({'cases': self.cases}, f)
            
    def load_library(self, path=None):
        """
        Load the case library.
        
        Parameters:
        -----------
        path : str or None
            Path to load the library from. If None, uses the library_path.
        """
        if path is None:
            path = self.library_path
            
        if path is None or not os.path.exists(path):
            raise ValueError(f"Library file not found: {path}")
            
        # Load library
        with open(path, 'r') as f:
            data = json.load(f)
            self.cases = data['cases']
            
    def get_library_stats(self):
        """
        Get statistics about the case library.
        
        Returns:
        --------
        stats : dict
            Dictionary containing library statistics.
        """
        # Count cases by diagnosis
        diagnosis_counts = {}
        for case in self.cases:
            diagnosis = case['diagnosis']
            if diagnosis in diagnosis_counts:
                diagnosis_counts[diagnosis] += 1
            else:
                diagnosis_counts[diagnosis] = 1
                
        # Count cases by difficulty
        difficulty_counts = {
            'easy': 0,
            'medium': 0,
            'hard': 0
        }
        
        for case in self.cases:
            difficulty = case['metadata'].get('difficulty', 'medium')
            if difficulty in difficulty_counts:
                difficulty_counts[difficulty] += 1
                
        # Compile stats
        stats = {
            'total_cases': len(self.cases),
            'diagnosis_distribution': diagnosis_counts,
            'difficulty_distribution': difficulty_counts
        }
        
        return stats
    
    def plot_library_stats(self):
        """
        Plot statistics about the case library.
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get stats
        stats = self.get_library_stats()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot diagnosis distribution
        diagnosis_labels = list(stats['diagnosis_distribution'].keys())
        diagnosis_values = list(stats['diagnosis_distribution'].values())
        
        ax1.bar(diagnosis_labels, diagnosis_values)
        ax1.set_title('Cases by Diagnosis')
        ax1.set_xlabel('Diagnosis')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot difficulty distribution
        difficulty_labels = list(stats['difficulty_distribution'].keys())
        difficulty_values = list(stats['difficulty_distribution'].values())
        
        ax2.bar(difficulty_labels, difficulty_values)
        ax2.set_title('Cases by Difficulty')
        ax2.set_xlabel('Difficulty')
        ax2.set_ylabel('Count')
        
        fig.tight_layout()
        return fig


class QuizGenerator:
    """
    Class for generating ECG interpretation quizzes.
    """
    
    def __init__(self, case_library):
        """
        Initialize the quiz generator.
        
        Parameters:
        -----------
        case_library : CaseLibrary
            Case library to use for quiz generation.
        """
        self.case_library = case_library
        
    def generate_quiz(self, n_questions=10, difficulty=None, diagnoses=None):
        """
        Generate a quiz.
        
        Parameters:
        -----------
        n_questions : int
            Number of questions. Default is 10.
        difficulty : str or None
            Difficulty level. If None, selects from any difficulty.
        diagnoses : list or None
            List of diagnoses to include. If None, selects from any diagnosis.
            
        Returns:
        --------
        quiz : dict
            Dictionary containing quiz data.
        """
        # Filter cases
        filtered_cases = self.case_library.cases
        
        if difficulty is not None:
            filtered_cases = [case for case in filtered_cases if case['metadata']['difficulty'] == difficulty]
            
        if diagnoses is not None:
            filtered_cases = [case for case in filtered_cases if case['diagnosis'] in diagnoses]
            
        if len(filtered_cases) < n_questions:
            raise ValueError(f"Not enough cases ({len(filtered_cases)}) for requested quiz size ({n_questions}).")
            
        # Select random cases
        selected_cases = random.sample(filtered_cases, n_questions)
        
        # Create quiz
        quiz = {
            'id': f"quiz_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'n_questions': n_questions,
            'difficulty': difficulty,
            'questions': []
        }
        
        # Add questions
        for i, case in enumerate(selected_cases):
            # Get all unique diagnoses for options
            all_diagnoses = list(set(c['diagnosis'] for c in self.case_library.cases))
            
            # Ensure correct answer is in options
            correct_diagnosis = case['diagnosis']
            
            # Select random incorrect options
            incorrect_options = [d for d in all_diagnoses if d != correct_diagnosis]
            incorrect_options = random.sample(incorrect_options, min(3, len(incorrect_options)))
            
            # Combine options and shuffle
            options = [correct_diagnosis] + incorrect_options
            random.shuffle(options)
            
            # Create question
            question = {
                'id': f"q{i+1}",
                'case_id': case['id'],
                'options': options,
                'correct_answer': correct_diagnosis
            }
            
            quiz['questions'].append(question)
            
        return quiz
    
    def save_quiz(self, quiz, path):
        """
        Save a quiz to file.
        
        Parameters:
        -----------
        quiz : dict
            Quiz data.
        path : str
            Path to save the quiz.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save quiz
        with open(path, 'w') as f:
            json.dump(quiz, f)
            
    def load_quiz(self, path):
        """
        Load a quiz from file.
        
        Parameters:
        -----------
        path : str
            Path to load the quiz from.
            
        Returns:
        --------
        quiz : dict
            Quiz data.
        """
        # Load quiz
        with open(path, 'r') as f:
            quiz = json.load(f)
            
        return quiz


class QuizSession:
    """
    Class for managing an ECG interpretation quiz session.
    """
    
    def __init__(self, quiz, case_library):
        """
        Initialize the quiz session.
        
        Parameters:
        -----------
        quiz : dict
            Quiz data.
        case_library : CaseLibrary
            Case library containing the quiz cases.
        """
        self.quiz = quiz
        self.case_library = case_library
        self.current_question_idx = 0
        self.answers = {}
        self.start_time = datetime.now()
        self.end_time = None
        
    def get_current_question(self):
        """
        Get the current question.
        
        Returns:
        --------
        question : dict
            Current question data.
        """
        if self.current_question_idx >= len(self.quiz['questions']):
            return None
            
        question = self.quiz['questions'][self.current_question_idx]
        
        # Get case data
        case = self.case_library.get_case(question['case_id'])
        
        # Create question with case data
        question_with_case = {
            'id': question['id'],
            'ecg_data': case['ecg_data'],
            'options': question['options'],
            'metadata': case['metadata']
        }
        
        return question_with_case
    
    def answer_question(self, answer):
        """
        Answer the current question.
        
        Parameters:
        -----------
        answer : str
            Selected answer.
            
        Returns:
        --------
        is_correct : bool
            Whether the answer is correct.
        """
        if self.current_question_idx >= len(self.quiz['questions']):
            raise ValueError("No more questions in quiz.")
            
        question = self.quiz['questions'][self.current_question_idx]
        correct_answer = question['correct_answer']
        
        # Record answer
        self.answers[question['id']] = {
            'selected': answer,
            'correct': correct_answer,
            'is_correct': answer == correct_answer,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Move to next question
        self.current_question_idx += 1
        
        # Check if quiz is complete
        if self.current_question_idx >= len(self.quiz['questions']):
            self.end_time = datetime.now()
            
        return answer == correct_answer
    
    def get_results(self):
        """
        Get quiz results.
        
        Returns:
        --------
        results : dict
            Dictionary containing quiz results.
        """
        # Calculate score
        n_correct = sum(1 for a in self.answers.values() if a['is_correct'])
        score = n_correct / len(self.quiz['questions']) if self.quiz['questions'] else 0
        
        # Calculate time taken
        if self.end_time is None:
            time_taken = (datetime.now() - self.start_time).total_seconds()
        else:
            time_taken = (self.end_time - self.start_time).total_seconds()
            
        # Compile results
        results = {
            'quiz_id': self.quiz['id'],
            'n_questions': len(self.quiz['questions']),
            'n_answered': len(self.answers),
            'n_correct': n_correct,
            'score': score,
            'time_taken': time_taken,
            'answers': self.answers
        }
        
        return results
    
    def get_feedback(self):
        """
        Get feedback for answered questions.
        
        Returns:
        --------
        feedback : list
            List of feedback for each answered question.
        """
        feedback = []
        
        for question in self.quiz['questions']:
            question_id = question['id']
            
            if question_id in self.answers:
                # Get case
                case = self.case_library.get_case(question['case_id'])
                
                # Get answer
                answer = self.answers[question_id]
                
                # Create feedback
                question_feedback = {
                    'question_id': question_id,
                    'case_id': question['case_id'],
                    'ecg_data': case['ecg_data'],
                    'selected_answer': answer['selected'],
                    'correct_answer': answer['correct'],
                    'is_correct': answer['is_correct'],
                    'explanation': case['metadata'].get('explanation', 'No explanation available.')
                }
                
                feedback.append(question_feedback)
                
        return feedback
    
    def plot_results(self):
        """
        Plot quiz results.
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get results
        results = self.get_results()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot score
        ax1.bar(['Correct', 'Incorrect'], [results['n_correct'], results['n_questions'] - results['n_correct']])
        ax1.set_title(f"Quiz Score: {results['score'] * 100:.1f}%")
        ax1.set_ylabel('Number of Questions')
        
        # Plot performance by diagnosis
        diagnosis_results = {}
        
        for question in self.quiz['questions']:
            question_id = question['id']
            
            if question_id in self.answers:
                # Get case
                case = self.case_library.get_case(question['case_id'])
                diagnosis = case['diagnosis']
                
                # Get answer
                answer = self.answers[question_id]
                
                # Add to diagnosis results
                if diagnosis not in diagnosis_results:
                    diagnosis_results[diagnosis] = {'correct': 0, 'total': 0}
                    
                diagnosis_results[diagnosis]['total'] += 1
                if answer['is_correct']:
                    diagnosis_results[diagnosis]['correct'] += 1
                    
        # Plot diagnosis performance
        diagnoses = list(diagnosis_results.keys())
        performance = [results['correct'] / results['total'] * 100 for results in diagnosis_results.values()]
        
        ax2.bar(diagnoses, performance)
        ax2.set_title('Performance by Diagnosis')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        fig.tight_layout()
        return fig


class PerformanceTracker:
    """
    Class for tracking user performance in ECG interpretation.
    """
    
    def __init__(self, user_id, data_path=None):
        """
        Initialize the performance tracker.
        
        Parameters:
        -----------
        user_id : str
            User ID.
        data_path : str or None
            Path to save performance data. If None, doesn't save data.
        """
        self.user_id = user_id
        self.data_path = data_path
        self.quiz_results = []
        self.performance_by_diagnosis = {}
        self.performance_by_difficulty = {}
        
        # Load data if available
        if data_path is not None and os.path.exists(data_path):
            self.load_data(data_path)
            
    def add_quiz_result(self, quiz_result):
        """
        Add a quiz result.
        
        Parameters:
        -----------
        quiz_result : dict
            Quiz result data.
        """
        # Add timestamp if not present
        if 'timestamp' not in quiz_result:
            quiz_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # Add quiz result
        self.quiz_results.append(quiz_result)
        
        # Update performance by diagnosis
        for question_id, answer in quiz_result['answers'].items():
            diagnosis = answer['correct']
            
            if diagnosis not in self.performance_by_diagnosis:
                self.performance_by_diagnosis[diagnosis] = {'correct': 0, 'total': 0}
                
            self.performance_by_diagnosis[diagnosis]['total'] += 1
            if answer['is_correct']:
                self.performance_by_diagnosis[diagnosis]['correct'] += 1
                
        # Save data
        if self.data_path is not None:
            self.save_data()
            
    def get_overall_performance(self):
        """
        Get overall performance statistics.
        
        Returns:
        --------
        stats : dict
            Dictionary containing performance statistics.
        """
        # Calculate overall accuracy
        total_questions = sum(result['n_questions'] for result in self.quiz_results)
        total_correct = sum(result['n_correct'] for result in self.quiz_results)
        
        accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        # Calculate average score
        avg_score = np.mean([result['score'] for result in self.quiz_results]) if self.quiz_results else 0
        
        # Calculate performance trend
        scores = [(result['timestamp'], result['score']) for result in self.quiz_results]
        scores.sort(key=lambda x: x[0])
        
        trend = [score for _, score in scores]
        
        # Compile stats
        stats = {
            'total_quizzes': len(self.quiz_results),
            'total_questions': total_questions,
            'total_correct': total_correct,
            'accuracy': accuracy,
            'avg_score': avg_score,
            'score_trend': trend,
            'performance_by_diagnosis': self.performance_by_diagnosis
        }
        
        return stats
    
    def get_diagnosis_performance(self, diagnosis):
        """
        Get performance statistics for a specific diagnosis.
        
        Parameters:
        -----------
        diagnosis : str
            Diagnosis to get statistics for.
            
        Returns:
        --------
        stats : dict
            Dictionary containing performance statistics.
        """
        if diagnosis not in self.performance_by_diagnosis:
            return {
                'diagnosis': diagnosis,
                'total': 0,
                'correct': 0,
                'accuracy': 0
            }
            
        performance = self.performance_by_diagnosis[diagnosis]
        
        stats = {
            'diagnosis': diagnosis,
            'total': performance['total'],
            'correct': performance['correct'],
            'accuracy': performance['correct'] / performance['total'] if performance['total'] > 0 else 0
        }
        
        return stats
    
    def get_improvement_suggestions(self):
        """
        Get suggestions for improvement.
        
        Returns:
        --------
        suggestions : list
            List of improvement suggestions.
        """
        suggestions = []
        
        # Check if enough data
        if len(self.quiz_results) < 2:
            suggestions.append("Complete more quizzes to get personalized improvement suggestions.")
            return suggestions
            
        # Find diagnoses with low accuracy
        for diagnosis, performance in self.performance_by_diagnosis.items():
            accuracy = performance['correct'] / performance['total'] if performance['total'] > 0 else 0
            
            if accuracy < 0.6 and performance['total'] >= 3:
                suggestions.append(f"Focus on improving recognition of {diagnosis}. Current accuracy: {accuracy * 100:.1f}%")
                
        # Check for improvement over time
        scores = [(result['timestamp'], result['score']) for result in self.quiz_results]
        scores.sort(key=lambda x: x[0])
        
        if len(scores) >= 5:
            recent_scores = [score for _, score in scores[-5:]]
            early_scores = [score for _, score in scores[:5]]
            
            if np.mean(recent_scores) <= np.mean(early_scores):
                suggestions.append("Your recent performance has not improved. Consider reviewing ECG interpretation fundamentals.")
                
        # Add general suggestions if no specific ones
        if not suggestions:
            suggestions.append("Your performance is consistent. Try increasing the difficulty level to challenge yourself.")
            
        return suggestions
    
    def save_data(self, path=None):
        """
        Save performance data.
        
        Parameters:
        -----------
        path : str or None
            Path to save the data. If None, uses the data_path.
        """
        if path is None:
            path = self.data_path
            
        if path is None:
            raise ValueError("No path specified for saving data.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save data
        data = {
            'user_id': self.user_id,
            'quiz_results': self.quiz_results,
            'performance_by_diagnosis': self.performance_by_diagnosis
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
            
    def load_data(self, path=None):
        """
        Load performance data.
        
        Parameters:
        -----------
        path : str or None
            Path to load the data from. If None, uses the data_path.
        """
        if path is None:
            path = self.data_path
            
        if path is None or not os.path.exists(path):
            raise ValueError(f"Data file not found: {path}")
            
        # Load data
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.user_id = data['user_id']
        self.quiz_results = data['quiz_results']
        self.performance_by_diagnosis = data['performance_by_diagnosis']
        
    def plot_performance_trend(self):
        """
        Plot performance trend.
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get scores
        scores = [(result['timestamp'], result['score']) for result in self.quiz_results]
        scores.sort(key=lambda x: x[0])
        
        timestamps = [ts for ts, _ in scores]
        score_values = [score for _, score in scores]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot trend
        ax.plot(range(len(score_values)), score_values, 'o-')
        ax.set_title('Performance Trend')
        ax.set_xlabel('Quiz Number')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1.1])
        ax.grid(True)
        
        # Add trend line
        if len(score_values) >= 2:
            z = np.polyfit(range(len(score_values)), score_values, 1)
            p = np.poly1d(z)
            ax.plot(range(len(score_values)), p(range(len(score_values))), 'r--', label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()
            
        fig.tight_layout()
        return fig
    
    def plot_diagnosis_performance(self):
        """
        Plot performance by diagnosis.
        
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Get diagnosis performance
        diagnoses = list(self.performance_by_diagnosis.keys())
        accuracies = []
        
        for diagnosis in diagnoses:
            performance = self.performance_by_diagnosis[diagnosis]
            accuracy = performance['correct'] / performance['total'] if performance['total'] > 0 else 0
            accuracies.append(accuracy)
            
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)
        diagnoses = [diagnoses[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot performance
        ax.barh(diagnoses, [acc * 100 for acc in accuracies])
        ax.set_title('Performance by Diagnosis')
        ax.set_xlabel('Accuracy (%)')
        ax.set_xlim([0, 105])
        
        # Add values
        for i, acc in enumerate(accuracies):
            ax.text(acc * 100 + 2, i, f"{acc * 100:.1f}%")
            
        fig.tight_layout()
        return fig


class TeachingSystem:
    """
    Class for managing the ECG teaching system.
    """
    
    def __init__(self, case_library_path=None, model_path=None):
        """
        Initialize the teaching system.
        
        Parameters:
        -----------
        case_library_path : str or None
            Path to the case library. If None, creates a new library.
        model_path : str or None
            Path to the AI model. If None, doesn't use AI assistance.
        """
        # Initialize case library
        self.case_library = CaseLibrary(case_library_path)
        
        # Initialize quiz generator
        self.quiz_generator = QuizGenerator(self.case_library)
        
        # Load AI model if provided
        self.model = None
        if model_path is not None and os.path.exists(model_path):
            self.model = load_model(model_path)
            
        # Initialize user performance trackers
        self.user_trackers = {}
        
    def create_quiz(self, n_questions=10, difficulty=None, diagnoses=None):
        """
        Create a new quiz.
        
        Parameters:
        -----------
        n_questions : int
            Number of questions. Default is 10.
        difficulty : str or None
            Difficulty level. If None, selects from any difficulty.
        diagnoses : list or None
            List of diagnoses to include. If None, selects from any diagnosis.
            
        Returns:
        --------
        quiz : dict
            Quiz data.
        """
        return self.quiz_generator.generate_quiz(n_questions, difficulty, diagnoses)
    
    def start_quiz_session(self, quiz):
        """
        Start a new quiz session.
        
        Parameters:
        -----------
        quiz : dict
            Quiz data.
            
        Returns:
        --------
        session : QuizSession
            Quiz session.
        """
        return QuizSession(quiz, self.case_library)
    
    def get_user_tracker(self, user_id, data_path=None):
        """
        Get a user performance tracker.
        
        Parameters:
        -----------
        user_id : str
            User ID.
        data_path : str or None
            Path to save performance data. If None, doesn't save data.
            
        Returns:
        --------
        tracker : PerformanceTracker
            User performance tracker.
        """
        if user_id not in self.user_trackers:
            self.user_trackers[user_id] = PerformanceTracker(user_id, data_path)
            
        return self.user_trackers[user_id]
    
    def get_ai_diagnosis(self, ecg_data):
        """
        Get AI diagnosis for ECG data.
        
        Parameters:
        -----------
        ecg_data : array-like
            ECG data.
            
        Returns:
        --------
        diagnosis : str
            AI diagnosis.
        confidence : float
            Confidence score.
        """
        if self.model is None:
            raise ValueError("No AI model loaded.")
            
        # Preprocess ECG data
        # This depends on the model's expected input format
        # Here we assume the model expects a batch of ECG signals
        if len(ecg_data.shape) == 2:
            # Add batch dimension
            ecg_data = np.expand_dims(ecg_data, axis=0)
            
        # Get prediction
        prediction = self.model.predict(ecg_data)
        
        # Get diagnosis and confidence
        diagnosis_idx = np.argmax(prediction[0])
        confidence = prediction[0][diagnosis_idx]
        
        # Map index to diagnosis
        # This depends on the model's output format
        # Here we assume the model outputs probabilities for each class
        diagnoses = ['Normal', 'Atrial Fibrillation', 'First-degree AV Block', 'Left Bundle Branch Block', 'Right Bundle Branch Block']
        
        if diagnosis_idx < len(diagnoses):
            diagnosis = diagnoses[diagnosis_idx]
        else:
            diagnosis = f"Class {diagnosis_idx}"
            
        return diagnosis, confidence
    
    def compare_user_ai_performance(self, user_id, n_cases=50):
        """
        Compare user performance with AI.
        
        Parameters:
        -----------
        user_id : str
            User ID.
        n_cases : int
            Number of cases to use for comparison. Default is 50.
            
        Returns:
        --------
        comparison : dict
            Dictionary containing comparison results.
        """
        if self.model is None:
            raise ValueError("No AI model loaded.")
            
        # Get user tracker
        tracker = self.get_user_tracker(user_id)
        
        # Get user performance
        user_stats = tracker.get_overall_performance()
        
        # Select random cases
        if len(self.case_library.cases) < n_cases:
            n_cases = len(self.case_library.cases)
            
        selected_cases = random.sample(self.case_library.cases, n_cases)
        
        # Evaluate AI on selected cases
        ai_correct = 0
        user_correct_by_case = {}
        
        for case in selected_cases:
            # Get case data
            case_data = self.case_library.get_case(case['id'])
            ecg_data = case_data['ecg_data']
            true_diagnosis = case_data['diagnosis']
            
            # Get AI diagnosis
            ai_diagnosis, ai_confidence = self.get_ai_diagnosis(ecg_data)
            
            # Check if correct
            if ai_diagnosis == true_diagnosis:
                ai_correct += 1
                
            # Check if user has answered this case
            user_correct_by_case[case['id']] = False
            
            for quiz_result in tracker.quiz_results:
                for question_id, answer in quiz_result['answers'].items():
                    # Find the question in the quiz
                    for question in quiz_result['questions']:
                        if question['id'] == question_id and question['case_id'] == case['id']:
                            user_correct_by_case[case['id']] = answer['is_correct']
                            break
                            
        # Calculate AI accuracy
        ai_accuracy = ai_correct / n_cases
        
        # Calculate user accuracy on the same cases
        user_correct = sum(1 for correct in user_correct_by_case.values() if correct)
        user_answered = sum(1 for correct in user_correct_by_case.values() if correct is not None)
        
        user_accuracy_on_cases = user_correct / user_answered if user_answered > 0 else 0
        
        # Compile comparison
        comparison = {
            'n_cases': n_cases,
            'ai_accuracy': ai_accuracy,
            'user_overall_accuracy': user_stats['accuracy'],
            'user_accuracy_on_cases': user_accuracy_on_cases,
            'user_answered_cases': user_answered
        }
        
        return comparison
    
    def plot_user_ai_comparison(self, comparison):
        """
        Plot user vs AI performance comparison.
        
        Parameters:
        -----------
        comparison : dict
            Dictionary containing comparison results.
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot comparison
        labels = ['AI', 'User (Overall)', 'User (Same Cases)']
        accuracies = [
            comparison['ai_accuracy'] * 100,
            comparison['user_overall_accuracy'] * 100,
            comparison['user_accuracy_on_cases'] * 100
        ]
        
        ax.bar(labels, accuracies)
        ax.set_title('User vs AI Performance Comparison')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim([0, 105])
        
        # Add values
        for i, acc in enumerate(accuracies):
            ax.text(i, acc + 2, f"{acc:.1f}%", ha='center')
            
        fig.tight_layout()
        return fig
    
    def generate_learning_path(self, user_id):
        """
        Generate a personalized learning path for a user.
        
        Parameters:
        -----------
        user_id : str
            User ID.
            
        Returns:
        --------
        learning_path : dict
            Dictionary containing learning path data.
        """
        # Get user tracker
        tracker = self.get_user_tracker(user_id)
        
        # Get user performance
        user_stats = tracker.get_overall_performance()
        
        # Get improvement suggestions
        suggestions = tracker.get_improvement_suggestions()
        
        # Identify weak areas
        weak_areas = []
        
        for diagnosis, performance in tracker.performance_by_diagnosis.items():
            accuracy = performance['correct'] / performance['total'] if performance['total'] > 0 else 0
            
            if accuracy < 0.6 and performance['total'] >= 3:
                weak_areas.append({
                    'diagnosis': diagnosis,
                    'accuracy': accuracy,
                    'n_cases': performance['total']
                })
                
        # Sort weak areas by accuracy
        weak_areas.sort(key=lambda x: x['accuracy'])
        
        # Generate learning path
        learning_path = {
            'user_id': user_id,
            'current_level': 'beginner' if user_stats['accuracy'] < 0.6 else ('intermediate' if user_stats['accuracy'] < 0.8 else 'advanced'),
            'weak_areas': weak_areas,
            'suggestions': suggestions,
            'recommended_quizzes': []
        }
        
        # Add recommended quizzes
        if weak_areas:
            # Focus on weak areas
            for area in weak_areas[:3]:
                learning_path['recommended_quizzes'].append({
                    'title': f"Improve {area['diagnosis']} Recognition",
                    'n_questions': 10,
                    'difficulty': 'medium',
                    'diagnoses': [area['diagnosis']]
                })
        else:
            # General improvement
            learning_path['recommended_quizzes'].append({
                'title': 'General ECG Interpretation',
                'n_questions': 10,
                'difficulty': 'medium',
                'diagnoses': None
            })
            
        # Add challenge quiz
        learning_path['recommended_quizzes'].append({
            'title': 'Challenge Quiz',
            'n_questions': 5,
            'difficulty': 'hard',
            'diagnoses': None
        })
        
        return learning_path


# Example usage
if __name__ == "__main__":
    # Create a case library
    library = CaseLibrary()
    
    # Add some sample cases
    for i in range(10):
        # Generate synthetic ECG data
        ecg_data = np.random.randn(5000, 12)
        
        # Add case
        library.add_case(
            ecg_data=ecg_data,
            diagnosis=random.choice(['Normal', 'Atrial Fibrillation', 'First-degree AV Block']),
            difficulty=random.choice(['easy', 'medium', 'hard']),
            metadata={'patient_age': random.randint(20, 80)}
        )
        
    print(f"Created case library with {len(library.cases)} cases")
    
    # Create a quiz generator
    quiz_generator = QuizGenerator(library)
    
    # Generate a quiz
    quiz = quiz_generator.generate_quiz(n_questions=5)
    
    print(f"Generated quiz with {len(quiz['questions'])} questions")
    
    # Create a quiz session
    session = QuizSession(quiz, library)
    
    # Simulate answering questions
    for i in range(len(quiz['questions'])):
        question = session.get_current_question()
        
        if question:
            # Randomly select an answer
            answer = random.choice(question['options'])
            
            # Answer the question
            is_correct = session.answer_question(answer)
            
            print(f"Question {i+1}: {'Correct' if is_correct else 'Incorrect'}")
            
    # Get results
    results = session.get_results()
    
    print(f"Quiz score: {results['score'] * 100:.1f}%")
    
    # Create a performance tracker
    tracker = PerformanceTracker('user1')
    
    # Add quiz result
    tracker.add_quiz_result(results)
    
    # Get performance stats
    stats = tracker.get_overall_performance()
    
    print(f"Overall accuracy: {stats['accuracy'] * 100:.1f}%")
    
    # Get improvement suggestions
    suggestions = tracker.get_improvement_suggestions()
    
    print("Improvement suggestions:")
    for suggestion in suggestions:
        print(f"- {suggestion}")

