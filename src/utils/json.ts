export function extractJsonFromText(text: string): any {
  if (!text) {
    console.warn("Empty input text. Returning empty object.");
    return {}; // Safe fallback
  }

  try {
    // Regex to extract JSON object (non-greedy match)
    const jsonRegex = /\{[\s\S]*?\}/;
    const match = text.match(jsonRegex);

    if (match && match[0]) {
      const jsonString = match[0];
      try {
        return JSON.parse(jsonString);
      } catch (jsonParseError) {
        console.error("Error parsing JSON string:", jsonParseError);
        console.log("Received text:", jsonString);
        return {}; // Safe fallback instead of null
      }
    } else {
      console.warn("No JSON object found in text using regex. Returning empty object.");
      console.log("Received text:", text);
      return {}; // Safe fallback
    }
  } catch (regexError) {
    console.error("Regex error during JSON extraction:", regexError);
    console.log("Received text:", text);
    return {}; // Safe fallback
  }
}


export function isValidJSON(jsonString: string): boolean {
  try {
    JSON.parse(jsonString);
    return true; // Parsing succeeded, it's valid JSON
  } catch {
    return false; // Parsing failed, it's not valid JSON
  }
}

export function safeParseJSON<T>(jsonString: string, defaultValue: T): T {
  try {
    return JSON.parse(jsonString) as T; // Attempt to parse and cast to type T
  } catch {
    return defaultValue; // Return defaultValue if parsing fails
  }
}

export function stringifyJSON(jsonObject: any, prettyPrint: boolean = false): string | null {
  try {
    if (prettyPrint) {
      return JSON.stringify(jsonObject, null, 2); // Pretty print with indentation of 2 spaces
    } else {
      return JSON.stringify(jsonObject);
    }
  } catch {
    return null; // Return null if stringification fails
  }
}
