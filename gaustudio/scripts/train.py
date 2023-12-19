from gaustudio.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)

def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

if __name__ == "__main__":
    main()