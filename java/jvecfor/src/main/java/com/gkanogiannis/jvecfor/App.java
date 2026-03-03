package com.gkanogiannis.jvecfor;

import com.gkanogiannis.jvecfor.cli.MainCommand;
import picocli.CommandLine;

/** The main entry point for the application. */
public final class App {

    private App() {}

    /**
     * The main method that serves as the application's entry point.
     *
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        int exitCode = run(args);
        System.exit(exitCode);
    }

    /**
     * Runs the application logic without exiting the JVM. This is the primary method to be used for
     * testing.
     *
     * @param args Command line arguments.
     * @return The exit code of the application.
     */
    public static int run(String[] args) {
        return new CommandLine(new MainCommand()).execute(args);
    }
}
