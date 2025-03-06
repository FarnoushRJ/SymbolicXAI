import click, ast
# Preliminary functions
class PythonLiteralOption(click.Option):
    def type_cast_value(self, ctx, value):
        # Either we denote a range, or a list with precise samples:
        # 1) Range:
        if 'range' in value:
            idx_open = value.find('(')
            idx_close = value.find(')')
            # Get the range input
            range_input = value[idx_open + 1:idx_close].split(',')
            # Make it to numbers:
            range_input = [int(num) for num in range_input]
            try :
                return list(range(*range_input))
            except:
                raise click.BadParameter(value)
        else:
            try:
                return ast.literal_eval(value)
            except:
                raise click.BadParameter(value)
