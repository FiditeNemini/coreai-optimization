{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autopydantic_model:: {{ objname }}

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block presets %}
   {% if has_presets(fullname) %}
   .. toctree::
      :hidden:

      /api/generated/{{ fullname }}.presets
   {% endif %}
   {% endblock %}
