package com.gkanogiannis.jvecfor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class AppTest {

    @Test
    void helpOptionExitsZero() {
        int exitCode = App.run(new String[] {"--help"});
        assertEquals(0, exitCode, "Help option should exit with code 0.");
    }

    @Test
    void unknownOptionFails() {
        int exitCode = App.run(new String[] {"--unknown-option"});
        assertNotEquals(0, exitCode, "Unknown option should fail.");
    }

    @Test
    void stdinMode_validData_exitsZero() throws Exception {
        String tsv = "1.0\t2.0\n3.0\t4.0\n5.0\t6.0\n7.0\t8.0\n";
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(tsv.getBytes(StandardCharsets.UTF_8)));
            int exitCode = App.run(new String[] {"-k", "2"});
            assertEquals(0, exitCode, "Valid stdin data should exit 0.");
        } finally {
            System.setIn(original);
        }
    }

    @Test
    void stdinMode_emptyInput_exitsNonZero() throws Exception {
        InputStream original = System.in;
        try {
            System.setIn(new ByteArrayInputStream(new byte[0]));
            int exitCode = App.run(new String[] {"-k", "2"});
            assertNotEquals(0, exitCode, "Empty stdin should fail.");
        } finally {
            System.setIn(original);
        }
    }
}
