package org.navneev.utils;

/**
 * Utility class for checking debug mode configuration.
 * Reads the "vector.debug" system property to enable debug output.
 */
public class EnvironmentUtils {

    private static final String DEBUG_CODE = "vector.debug";

    private static final String DEFAULT_DEBUG_CODE = "false";

    private static final boolean DEBUG = Boolean.parseBoolean(System.getProperty(DEBUG_CODE,
            DEFAULT_DEBUG_CODE));

    /**
     * Checks if debug mode is enabled via system property.
     * @return true if debug mode is enabled, false otherwise
     */
    public static boolean isDebug() {
        return DEBUG;
    }

}
