import * as fs from 'node:fs/promises';
import * as readline from 'node:readline';
import { research } from './deep-research.js';
import {
  writeFinalReport,
  type ResearchProgress,
} from './deep-research.js';
import { generateFeedback } from './feedback.js';
import { OutputManager } from './output-manager.js';

const output = new OutputManager();

// Helper function for consistent logging
function log(...args: unknown[]) {
  output.log(
    args.map(arg =>
      typeof arg === 'string' ? arg : JSON.stringify(arg)
    ).join(' ')
  );
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Helper function to get user input (Promise-based)
function askQuestion(query: string): Promise<string> {
  return new Promise(resolve => {
    rl.question(query, answer => {
      resolve(answer.trim()); // Trim whitespace from answer
    });
  });
}

// Run the agent - main function
async function run() {
  try {
    // Get initial query from command line arguments or user input
    let initialQuery: string;
    if (process.argv.length > 2) {
      initialQuery = process.argv.slice(2).join(' '); // Get query from command line args
      log(`Research query from command line: "${initialQuery}"`);
    } else {
      initialQuery = await askQuestion('What would you like to research? ');
    }

    if (!initialQuery) {
      log('Research query cannot be empty. Please provide a query.');
      rl.close();
      return; // Exit if no query is provided
    }

    // Get breadth and depth parameters with input validation and defaults
    let breadth: number;
    const breadthInput = await askQuestion(
      'Enter research breadth (recommended 2-10, default 4): ',
    );
    breadth = Number.parseInt(breadthInput, 10);
    breadth = isNaN(breadth) ? 4 : Math.max(1, Math.min(10, breadth)); // Default 4, range 1-10

    let depth: number;
    const depthInput = await askQuestion(
      'Enter research depth (recommended 1-5, default 2): ',
    );
    depth = Number.parseInt(depthInput, 10);
    depth = isNaN(depth) ? 2 : Math.max(1, Math.min(5, depth)); // Default 2, range 1-5

    log(`Research parameters: Breadth = ${breadth}, Depth = ${depth}`);
    log('Creating research plan...');

    // Generate follow-up questions
    const feedbackResult = await generateFeedback({
      query: initialQuery,
    });

    // Check if feedbackResult is null or undefined
    const followUpQuestions = feedbackResult?.followUpQuestions || [];

    let answers: string[] = [];

    if (followUpQuestions && followUpQuestions.length > 0) {
      log(
        '\nTo better understand your research needs, please answer these follow-up questions:',
      );

      // Collect answers to follow-up questions
      answers = [];
      for (const question of followUpQuestions) {
        const answer = await askQuestion(`\n${question}\nYour answer: `);
        answers.push(answer);
      }
    } else {
      log('\nNo follow-up questions needed. Proceeding with research.');
    }

    // Combine all information for deep research
    const combinedQuery = `
Initial Query: ${initialQuery}
${
      followUpQuestions && followUpQuestions.length > 0
        ? `Follow-up Questions and Answers:\n${followUpQuestions
            .map((q: string, i: number) => `Q: ${q}\nA: ${answers[i]}`)
            .join('\n')}`
        : ''
    }
`;

    log('\nResearching your topic...');
    log('\nStarting research with progress tracking...\n');

    const { learnings, visitedUrls } = await research({
      query: combinedQuery,
      breadth,
      depth,
      onProgress: (progress: ResearchProgress) => {
        output.updateProgress(progress);
      },
    });

    if (!learnings || learnings.length === 0) {
      log('\nNo significant learnings were found during the research.');
    } else {
      log(`\n\nLearnings:\n\n${learnings.join('\n')}`);
    }

    if (visitedUrls && visitedUrls.length > 0) {
      log(
        `\n\nVisited URLs (${visitedUrls.length}):\n\n${visitedUrls.join('\n')}`,
      );
    } else {
      log('\nNo URLs were visited during the research.');
    }

    log('Writing final report...');

    const report = await writeFinalReport({
      prompt: combinedQuery,
      learnings,
      visitedUrls,
    });

    // Save report to file
    const reportFilename = 'output.md';
    await fs.writeFile(reportFilename, report, 'utf-8');

    console.log(`\n\nFinal Report:\n\n${report}`);
    console.log(`\nReport has been saved to ${reportFilename}`);
  } catch (error) {
    console.error('Error during research process:', error); // More informative error log
    log(`\nAn error occurred during the research process: ${
      error instanceof Error ? error.message : String(error)
    }`); // Log error to output manager as well
  } finally {
    rl.close(); // Ensure readline interface is always closed
    log('Research process finished.'); // Indicate process completion
  }
}

// Start the research process
run().catch(error => {
  console.error('Unhandled error in run function:', error); // Catch any unhandled errors
});
