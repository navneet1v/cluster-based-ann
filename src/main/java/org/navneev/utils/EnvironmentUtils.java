package org.navneev.utils;

/**
 * Utility class for checking debug mode configuration. Reads the "vector.debug" system property to
 * enable debug output.
 */
public class EnvironmentUtils {

    private static final String DEBUG_CODE = "vector.debug";

    private static final String DEFAULT_DEBUG_CODE = "false";

    private static final boolean DEBUG =
            Boolean.parseBoolean(System.getProperty(DEBUG_CODE, DEFAULT_DEBUG_CODE));

    private static final String BUILD_INDEX = "index.build";

    private static final String DEFAULT_BUILD_INDEX = "true";

    private static final boolean BUILD =
            Boolean.parseBoolean(System.getProperty(BUILD_INDEX, DEFAULT_BUILD_INDEX));

    /**
     * Checks if debug mode is enabled via system property.
     *
     * @return true if debug mode is enabled, false otherwise
     */
    public static boolean isDebug() {
        return DEBUG;
    }

    public static boolean isBuild() {
        return BUILD;
    }
}
